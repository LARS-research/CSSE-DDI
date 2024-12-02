import argparse
from scipy.sparse.csgraph import shortest_path
import numpy as np
import pandas as pd
import torch
import dgl
import json
import logging
import logging.config
import os
import psutil
from statistics import mean
import pickle


def parse_arguments():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser(description='SEAL')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--hop', type=int, default=1)
    parser.add_argument('--model', type=str, default='dgcnn')
    parser.add_argument('--gcn_type', type=str, default='gcn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_units', type=int, default=32)
    parser.add_argument('--sort_k', type=int, default=30)
    parser.add_argument('--pooling', type=str, default='sum')
    parser.add_argument('--dropout', type=str, default=0.5)
    parser.add_argument('--hits_k', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--subsample_ratio', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--random_seed', type=int, default=2023)
    parser.add_argument('--save_dir', type=str, default='./processed')
    args = parser.parse_args()

    return args


def coalesce_graph(graph, aggr_type='sum', copy_data=False):
    """
    Coalesce multi-edge graph
    Args:
        graph(DGLGraph): graph
        aggr_type(str): type of aggregator for multi edge weights
        copy_data(bool): if copy ndata and edata in new graph

    Returns:
        graph(DGLGraph): graph


    """
    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df['edge_weight'] = graph.edata['edge_weight'].numpy()

    if aggr_type == 'sum':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].sum().reset_index()
    elif aggr_type == 'mean':
        tmp = graph_df.groupby(['src', 'dst'])['edge_weight'].mean().reset_index()
    else:
        raise ValueError("aggr type error")

    if copy_data:
        graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True)
    else:
        graph = dgl.to_simple(graph)

    src, dst = graph.edges()
    graph_df = pd.DataFrame({'src': src, 'dst': dst})
    graph_df = pd.merge(graph_df, tmp, how='left', on=['src', 'dst'])
    graph.edata['edge_weight'] = torch.from_numpy(graph_df['edge_weight'].values).unsqueeze(1)

    graph.edata.pop('count')
    return graph


def drnl_node_labeling(subgraph, src, dst):
    """
    Double Radius Node labeling
    d = r(i,u)+r(i,v)
    label = 1+ min(r(i,u),r(i,v))+ (d//2)*(d//2+d%2-1)
    Isolated nodes in subgraph will be set as zero.
    Extreme large graph may cause memory error.

    Args:
        subgraph(DGLGraph): The graph
        src(int): node id of one of src node in new subgraph
        dst(int): node id of one of dst node in new subgraph
    Returns:
        z(Tensor): node labeling tensor
    """
    adj = subgraph.adj().to_dense().numpy()
    src, dst = (dst, src) if src > dst else (src, dst)
    if src != dst:
        idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
        adj_wo_src = adj[idx, :][:, idx]

        idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
        dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
        dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)
    else:
        dist2src = shortest_path(adj, directed=False, unweighted=True, indices=src)
        # dist2src = np.insert(dist2src, dst, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(adj, directed=False, unweighted=True, indices=dst)
        # dist2dst = np.insert(dist2dst, src, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2

    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.

    return z.to(torch.long)


def cal_ranks(probs, label):
    sorted_idx = np.argsort(probs, axis=1)[:,::-1]
    find_target = sorted_idx == np.expand_dims(label, 1)
    ranks = np.nonzero(find_target)[1] + 1
    return ranks

def cal_performance(ranks):
    mrr = (1. / ranks).sum() / len(ranks)
    m_r = sum(ranks) * 1.0 / len(ranks)
    h_1 = sum(ranks<=1) * 1.0 / len(ranks)
    h_10 = sum(ranks<=10) * 1.0 / len(ranks)
    return mrr, m_r, h_1, h_10

def get_logger(name, log_dir):
    config_dict = json.load(open('./config/' + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name + '.log'
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)
    return logger

def get_current_memory_gb() -> int:
# 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. / 1024.

def get_f1_score_list(class_dict):
    f1_score_list = [[],[],[],[],[]]
    for key in class_dict:
        if key.isdigit():
            if class_dict[key]['support'] < 10:
                f1_score_list[0].append(class_dict[key]['f1-score'])
            elif 10 <= class_dict[key]['support'] < 50:
                f1_score_list[1].append(class_dict[key]['f1-score'])
            elif 50 <= class_dict[key]['support'] < 100:
                f1_score_list[2].append(class_dict[key]['f1-score'])
            elif 100 <= class_dict[key]['support'] < 1000:
                f1_score_list[3].append(class_dict[key]['f1-score'])
            elif 1000 <= class_dict[key]['support'] < 100000:
                f1_score_list[4].append(class_dict[key]['f1-score'])
    for index, _ in enumerate(f1_score_list):
        f1_score_list[index] = mean(_)
    return f1_score_list

def get_acc_list(class_dict):
    acc_list = [0.0,0.0,0.0,0.0,0.0]
    support_list = [0,0,0,0,0]
    proportion_list = [0.0,0.0,0.0,0.0,0.0]
    for key in class_dict:
        if key.isdigit():
            if class_dict[key]['support'] < 10:
                acc_list[0] += (class_dict[key]['recall']*class_dict[key]['support'])
                support_list[0]+=class_dict[key]['support']
            elif 10 <= class_dict[key]['support'] < 50:
                acc_list[1] += (class_dict[key]['recall']*class_dict[key]['support'])
                support_list[1] += class_dict[key]['support']
            elif 50 <= class_dict[key]['support'] < 100:
                acc_list[2] += (class_dict[key]['recall']*class_dict[key]['support'])
                support_list[2] += class_dict[key]['support']
            elif 100 <= class_dict[key]['support'] < 1000:
                acc_list[3] += (class_dict[key]['recall']*class_dict[key]['support'])
                support_list[3] += class_dict[key]['support']
            elif 1000 <= class_dict[key]['support'] < 100000:
                acc_list[4] += (class_dict[key]['recall'] * class_dict[key]['support'])
                support_list[4] += class_dict[key]['support']
    for index, _ in enumerate(acc_list):
        acc_list[index] = acc_list[index] / support_list[index]
        # proportion_list[index] = support_list[index] / class_dict['macro avg']['support']
    return acc_list

def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label', 'g_label', 'n_label')
    return dict(zip(keys, data_tuple))


class Temp_Scheduler(object):
    def __init__(self, total_epochs, curr_temp, base_temp, temp_min=0.33, last_epoch=-1):
        self.curr_temp = curr_temp
        self.base_temp = base_temp
        self.temp_min = temp_min
        self.last_epoch = last_epoch
        self.total_epochs = total_epochs
        self.step(last_epoch + 1)

    def step(self, epoch=None):
        return self.decay_whole_process()

    def decay_whole_process(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        # self.total_epochs = 150
        self.curr_temp = (1 - self.last_epoch / self.total_epochs) * (self.base_temp - self.temp_min) + self.temp_min
        if self.curr_temp < self.temp_min:
            self.curr_temp = self.temp_min
        return self.curr_temp