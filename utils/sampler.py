import os.path as osp
from tqdm import tqdm
from copy import deepcopy
import torch
import dgl
from torch.utils.data import DataLoader, Dataset
from dgl import DGLGraph, NID
from dgl.dataloading.negative_sampler import Uniform
from dgl import add_self_loop
import numpy as np
from scipy.sparse.csgraph import shortest_path
#Grail
from scipy.sparse import csc_matrix
import os
import json
import matplotlib.pyplot as plt
import logging
from scipy.special import softmax
from tqdm import tqdm
import lmdb
import multiprocessing as mp
import scipy.sparse as ssp
from utils.graph_utils import serialize, incidence_matrix, remove_nodes
from utils.dgl_utils import _bfs_relational
import struct


class GraphDataSet(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor, n_ent, n_rel, max_seq_length):
        self.graph_list = graph_list
        self.tensor = tensor
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        input_ids, input_mask, mask_position, mask_label = self.convert_subgraph_to_feature(self.graph_list[index],
                                                                                            self.max_seq_length,
                                                                                            self.tensor[index])
        rela_mat = torch.zeros(self.max_seq_length, self.max_seq_length, self.n_rel)
        rela_mat[self.graph_list[index].edges()[0], self.graph_list[index].edges()[1], self.graph_list[index].edata['rel']] = 1
        return (self.graph_list[index], self.tensor[index], input_ids, input_mask, mask_position, mask_label, rela_mat)

    def convert_subgraph_to_feature(self, subgraph, max_seq_length, triple):
        input_ids = [subgraph.ndata[NID]]
        input_mask = [1 for _ in range(subgraph.num_nodes())]
        if max_seq_length - subgraph.num_nodes() > 0:
            input_ids.append(torch.tensor([self.n_ent for _ in range(max_seq_length - subgraph.num_nodes())]))
            input_mask += [0 for _ in range(max_seq_length - subgraph.num_nodes())]
        input_ids = torch.cat(input_ids)
        input_mask = torch.tensor(input_mask)
        # input_ids = torch.cat(
        #     [subgraph.ndata[NID], torch.tensor([self.n_ent for _ in range(max_seq_length - subgraph.num_nodes())])])
        # input_mask = [1 for _ in range(subgraph.num_nodes())]
        # input_mask += [0 for _ in range(max_seq_length - subgraph.num_nodes())]
        # while len(input_mask) < max_seq_length:
        #     # input_ids.append(self.n_ent)
        #     input_mask.append(0)
        # TODO: predict head entity, now predict tail entity
        mask_position = ((subgraph.ndata[NID] == triple[2]).nonzero().flatten())
        input_ids[mask_position.item()] = self.n_ent + 1
        mask_label = triple[2]
        # for position in list(torch.where(subgraph.ndata['z']==1)[0]):
        #     mask_position = position
        #     input_ids[mask_position]=self.n_ent+1
        #     mask_label = subgraph.ndata[NID][mask_position]
        #     break
        assert input_ids.size(0) == max_seq_length
        assert input_mask.size(0) == max_seq_length

        return input_ids, input_mask, mask_position, mask_label


class GraphDataSetRP(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor, n_ent, n_rel, max_seq_length):
        self.graph_list = graph_list
        self.tensor = tensor
        self.n_ent = n_ent
        self.n_rel = n_rel
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        input_ids, num_nodes, head_position, tail_position = self.convert_subgraph_to_feature(self.graph_list[index],self.tensor[index],self.max_seq_length)
        rela_mat = torch.zeros(self.max_seq_length, self.max_seq_length, self.n_rel)
        rela_mat[self.graph_list[index].edges()[0], self.graph_list[index].edges()[1], self.graph_list[index].edata['rel']] = 1
        return (self.graph_list[index], self.tensor[index], input_ids, num_nodes, head_position, tail_position, rela_mat)

    def convert_subgraph_to_feature(self, subgraph, triple, max_seq_length):
        input_ids = [subgraph.ndata[NID]]
        if max_seq_length - subgraph.num_nodes() > 0:
            input_ids.append(torch.tensor([self.n_ent for _ in range(max_seq_length - subgraph.num_nodes())]))
        input_ids = torch.cat(input_ids)
        # print(subgraph)
        # print(input_ids)
        # print(triple)
        head_position = torch.where(subgraph.ndata[NID] == triple[0])[0]
        tail_position = torch.where(subgraph.ndata[NID] == triple[2])[0]
        # print(head_position)
        # print(tail_position)
        # exit(0)
        # input_ids = torch.cat(
        #     [subgraph.ndata[NID], torch.tensor([self.n_ent for _ in range(max_seq_length - subgraph.num_nodes())])])
        # input_mask = [1 for _ in range(subgraph.num_nodes())]
        # input_mask += [0 for _ in range(max_seq_length - subgraph.num_nodes())]
        # while len(input_mask) < max_seq_length:
        #     # input_ids.append(self.n_ent)
        #     input_mask.append(0)
        # for position in list(torch.where(subgraph.ndata['z']==1)[0]):
        #     mask_position = position
        #     input_ids[mask_position]=self.n_ent+1
        #     mask_label = subgraph.ndata[NID][mask_position]
        #     break
        assert input_ids.size(0) == max_seq_length

        return input_ids, subgraph.num_nodes(), head_position, tail_position

class GraphDataSetGCN(Dataset):
    """
    GraphDataset for torch DataLoader
    """

    def __init__(self, graph_list, tensor, n_ent, n_rel):
        self.graph_list = graph_list
        self.tensor = tensor

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, index):
        head_idx = torch.where(self.graph_list[index].ndata[NID] == self.tensor[index][0])[0]
        tail_idx = torch.where(self.graph_list[index].ndata[NID] == self.tensor[index][2])[0]
        return (self.graph_list[index], self.tensor[index], head_idx, tail_idx)

class PosNegEdgesGenerator(object):
    """
    Generate positive and negative samples
    Attributes:
        g(dgl.DGLGraph): graph
        split_edge(dict): split edge
        neg_samples(int): num of negative samples per positive sample
        subsample_ratio(float): ratio of subsample
        shuffle(bool): if shuffle generated graph list
    """

    def __init__(self, g, split_edge, neg_samples=1, subsample_ratio=0.1, shuffle=True):
        self.neg_sampler = Uniform(neg_samples)
        self.subsample_ratio = subsample_ratio
        self.split_edge = split_edge
        self.g = g
        self.shuffle = shuffle

    def __call__(self, split_type):

        if split_type == 'train':
            subsample_ratio = self.subsample_ratio
        else:
            subsample_ratio = 1

        pos_edges = self.g.edges()
        pos_edges = torch.stack((pos_edges[0], pos_edges[1]), 1)

        if split_type == 'train':
            # Adding self loop in train avoids sampling the source node itself.
            g = add_self_loop(self.g)
            eids = g.edge_ids(pos_edges[:, 0], pos_edges[:, 1])
            neg_edges = torch.stack(self.neg_sampler(g, eids), dim=1)
        else:
            neg_edges = self.split_edge[split_type]['edge_neg']
        pos_edges = self.subsample(pos_edges, subsample_ratio).long()
        neg_edges = self.subsample(neg_edges, subsample_ratio).long()

        edges = torch.cat([pos_edges, neg_edges])
        labels = torch.cat([torch.ones(pos_edges.size(0), 1), torch.zeros(neg_edges.size(0), 1)])
        if self.shuffle:
            perm = torch.randperm(edges.size(0))
            edges = edges[perm]
            labels = labels[perm]
        return edges, labels

    def subsample(self, edges, subsample_ratio):
        """
        Subsample generated edges.
        Args:
            edges(Tensor): edges to subsample
            subsample_ratio(float): ratio of subsample

        Returns:
            edges(Tensor):  edges

        """

        num_edges = edges.size(0)
        perm = torch.randperm(num_edges)
        perm = perm[:int(subsample_ratio * num_edges)]
        edges = edges[perm]
        return edges


class EdgeDataSet(Dataset):
    """
    Assistant Dataset for speeding up the SEALSampler
    """

    def __init__(self, triples, transform):
        self.transform = transform
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        edge = torch.tensor([self.triples[index]['triple'][0], self.triples[index]['triple'][2]])
        subgraph = self.transform(edge)
        return (subgraph)


class SEALSampler(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        num_workers(int): num of workers

    """

    def __init__(self, graph, hop=1, max_num_nodes=100, num_workers=32, type='greedy', print_fn=print):
        self.graph = graph
        self.hop = hop
        self.print_fn = print_fn
        self.num_workers = num_workers
        self.threshold = None
        self.max_num_nodes = max_num_nodes
        self.sample_type = type

    def sample_subgraph(self, target_nodes, mode='valid'):
        """
        Args:
            target_nodes(Tensor): Tensor of two target nodes
        Returns:
            subgraph(DGLGraph): subgraph
        """
        # TODO: Add sample constrain on each hop
        sample_nodes = [target_nodes]
        frontiers = target_nodes
        if self.sample_type == 'greedy':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                if tmp_sample_nodes.size(0) < self.max_num_nodes: # whether sample or not
                    frontiers = self.graph.out_edges(frontiers)[1]
                    frontiers = torch.unique(frontiers)
                    if frontiers.size(0) > self.max_num_nodes - tmp_sample_nodes.size(0):
                        frontiers = np.random.choice(frontiers.numpy(), self.max_num_nodes - tmp_sample_nodes.size(0), replace=False)
                        frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'greedy_set':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes_set = set(tmp_sample_nodes.tolist())
                # tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                if len(tmp_sample_nodes_set) < self.max_num_nodes: # whether sample or not
                    tmp_frontiers = self.graph.out_edges(frontiers)[1]
                    tmp_frontiers_set = set(tmp_frontiers.tolist())
                    tmp_frontiers_set = tmp_frontiers_set - tmp_sample_nodes_set
                    if not tmp_frontiers_set:
                        break
                    else:
                        frontiers_set = tmp_frontiers_set
                        frontiers = torch.tensor(list(frontiers_set))
                        if frontiers.size(0) > self.max_num_nodes - len(tmp_sample_nodes_set):
                            frontiers = np.random.choice(frontiers.numpy(), self.max_num_nodes - len(tmp_sample_nodes_set), replace=False)
                            frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'average_set':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes_set = set(tmp_sample_nodes.tolist())
                num_nodes = int(self.max_num_nodes / self.hop * (i + 1))
                if len(tmp_sample_nodes_set) < num_nodes: # whether sample or not
                    tmp_frontiers = self.graph.out_edges(frontiers)[1]
                    tmp_frontiers_set = set(tmp_frontiers.tolist())
                    tmp_frontiers_set = tmp_frontiers_set - tmp_sample_nodes_set
                    if not tmp_frontiers_set:
                        break
                    else:
                        frontiers_set = tmp_frontiers_set
                        frontiers = torch.tensor(list(frontiers_set))
                        if frontiers.size(0) > num_nodes - len(tmp_sample_nodes_set):
                            frontiers = np.random.choice(frontiers.numpy(), num_nodes - len(tmp_sample_nodes_set), replace=False)
                            frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'average':
            for i in range(self.hop):
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                num_nodes = int(self.max_num_nodes/self.hop*(i+1))
                if tmp_sample_nodes.size(0) < num_nodes:  # whether sample or not
                    frontiers = self.graph.out_edges(frontiers)[1]
                    frontiers = torch.unique(frontiers)
                    if frontiers.size(0) > num_nodes - tmp_sample_nodes.size(0):
                        frontiers = np.random.choice(frontiers.numpy(), num_nodes - tmp_sample_nodes.size(0),
                                                     replace=False)
                        frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'enclosing_subgraph':
            u, v = target_nodes[0], target_nodes[1]
            u_neighbour, v_neighbour = u, v
            # print(target_nodes)
            u_sample_nodes = [u_neighbour.reshape(1)]
            v_sample_nodes = [v_neighbour.reshape(1)]
            graph = self.graph
            if graph.has_edges_between(u, v):
                link_id = graph.edge_ids(u, v, return_uv=True)[2]
                graph.remove_edges(link_id)
            if graph.has_edges_between(v, u):
                link_id = graph.edge_ids(v, u, return_uv=True)[2]
                graph.remove_edges(link_id)
            for i in range(self.hop):
                u_frontiers = graph.out_edges(u_neighbour)[1]
                # v_frontiers = self.graph.out_edges(v_neighbour)[1]
                u_neighbour = u_frontiers
                # set(u_frontiers.tolist())
                if u_frontiers.size(0) > self.max_num_nodes:
                    u_frontiers = np.random.choice(u_frontiers.numpy(), self.max_num_nodes, replace=False)
                    u_frontiers = torch.tensor(u_frontiers)
                u_sample_nodes.append(u_frontiers)
            for i in range(self.hop):
                v_frontiers = graph.out_edges(v_neighbour)[1]
                # v_frontiers = self.graph.out_edges(v_neighbour)[1]
                v_neighbour = v_frontiers
                if v_frontiers.size(0) > self.max_num_nodes:
                    v_frontiers = np.random.choice(v_frontiers.numpy(), self.max_num_nodes, replace=False)
                    v_frontiers = torch.tensor(v_frontiers)
                v_sample_nodes.append(v_frontiers)
            # print('U', u_sample_nodes)
            # print('V', v_sample_nodes)
            u_sample_nodes = torch.cat(u_sample_nodes)
            u_sample_nodes = torch.unique(u_sample_nodes)
            v_sample_nodes = torch.cat(v_sample_nodes)
            v_sample_nodes = torch.unique(v_sample_nodes)
            # print('U', u_sample_nodes)
            # print('V', v_sample_nodes)
            u_sample_nodes_set = set(u_sample_nodes.tolist())
            v_sample_nodes_set = set(v_sample_nodes.tolist())
            uv_inter_neighbour = u_sample_nodes_set.intersection(v_sample_nodes_set)
            frontiers = torch.tensor(list(uv_inter_neighbour), dtype=torch.int64)
            # print(frontiers)
            sample_nodes.append(frontiers)
        else:
            raise NotImplementedError
        sample_nodes = torch.cat(sample_nodes)
        sample_nodes = torch.unique(sample_nodes)
        subgraph = dgl.node_subgraph(self.graph, sample_nodes)

        # Each node should have unique node id in the new subgraph
        u_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[0]), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[NID] == int(target_nodes[1]), as_tuple=False))
        # remove link between target nodes in positive subgraphs.
        # Edge removing will rearange NID and EID, which lose the original NID and EID.

        # if dgl.__version__[:5] < '0.6.0':
        #     nids = subgraph.ndata[NID]
        #     eids = subgraph.edata[EID]
        #     if subgraph.has_edges_between(u_id, v_id):
        #         link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
        #         subgraph.remove_edges(link_id)
        #         eids = eids[subgraph.edata[EID]]
        #     if subgraph.has_edges_between(v_id, u_id):
        #         link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
        #         subgraph.remove_edges(link_id)
        #         eids = eids[subgraph.edata[EID]]
        #     subgraph.ndata[NID] = nids
        #     subgraph.edata[EID] = eids

        if subgraph.has_edges_between(u_id, v_id):
            link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)
        if subgraph.has_edges_between(v_id, u_id):
            link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)

        z = drnl_node_labeling(subgraph, u_id, v_id)
        subgraph.ndata['z'] = z
        return subgraph

    def _collate(self, batch_graphs):

        # batch_graphs = map(list, zip(*batch))
        # print(batch_graphs)
        # print(batch_triples)
        # print(batch_labels)

        batch_graphs = dgl.batch(batch_graphs)
        return batch_graphs

    def __call__(self, triples):
        subgraph_list = []
        triples_list = []
        labels_list = []
        edge_dataset = EdgeDataSet(triples, transform=self.sample_subgraph)
        self.print_fn('Using {} workers in sampling job.'.format(self.num_workers))
        sampler = DataLoader(edge_dataset, batch_size=128, num_workers=self.num_workers,
                             shuffle=False, collate_fn=self._collate)
        for subgraph in tqdm(sampler, ncols=100):
            subgraph = dgl.unbatch(subgraph)

            subgraph_list += subgraph

        return subgraph_list


class SEALSampler_NG(object):
    """
    Sampler for SEAL in paper(no-block version)
    The  strategy is to sample all the k-hop neighbors around the two target nodes.
    Attributes:
        graph(DGLGraph): The graph
        hop(int): num of hop
        num_workers(int): num of workers

    """

    def __init__(self, graph, hop=1, max_num_nodes=100, num_workers=32, type='greedy', print_fn=print):
        self.graph = graph
        self.hop = hop
        self.print_fn = print_fn
        self.num_workers = num_workers
        self.threshold = None
        self.max_num_nodes = max_num_nodes
        self.sample_type = type

    def sample_subgraph(self, target_nodes, mode='valid'):
        """
        Args:
            target_nodes(Tensor): Tensor of two target nodes
        Returns:
            subgraph(DGLGraph): subgraph
        """
        # TODO: Add sample constrain on each hop
        sample_nodes = [target_nodes]
        frontiers = target_nodes
        if self.sample_type == 'greedy':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                if tmp_sample_nodes.size(0) < self.max_num_nodes: # whether sample or not
                    frontiers = self.graph.out_edges(frontiers)[1]
                    frontiers = torch.unique(frontiers)
                    if frontiers.size(0) > self.max_num_nodes - tmp_sample_nodes.size(0):
                        frontiers = np.random.choice(frontiers.numpy(), self.max_num_nodes - tmp_sample_nodes.size(0), replace=False)
                        frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'greedy_set':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes_set = set(tmp_sample_nodes.tolist())
                # tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                if len(tmp_sample_nodes_set) < self.max_num_nodes: # whether sample or not
                    tmp_frontiers = self.graph.out_edges(frontiers)[1]
                    tmp_frontiers_set = set(tmp_frontiers.tolist())
                    tmp_frontiers_set = tmp_frontiers_set - tmp_sample_nodes_set
                    if not tmp_frontiers_set:
                        break
                    else:
                        frontiers_set = tmp_frontiers_set
                        frontiers = torch.tensor(list(frontiers_set))
                        if frontiers.size(0) > self.max_num_nodes - len(tmp_sample_nodes_set):
                            frontiers = np.random.choice(frontiers.numpy(), self.max_num_nodes - len(tmp_sample_nodes_set), replace=False)
                            frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'average_set':
            for i in range(self.hop):
                # get sampled node number
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes_set = set(tmp_sample_nodes.tolist())
                num_nodes = int(self.max_num_nodes / self.hop * (i + 1))
                if len(tmp_sample_nodes_set) < num_nodes: # whether sample or not
                    tmp_frontiers = self.graph.out_edges(frontiers)[1]
                    tmp_frontiers_set = set(tmp_frontiers.tolist())
                    tmp_frontiers_set = tmp_frontiers_set - tmp_sample_nodes_set
                    if not tmp_frontiers_set:
                        break
                    else:
                        frontiers_set = tmp_frontiers_set
                        frontiers = torch.tensor(list(frontiers_set))
                        if frontiers.size(0) > num_nodes - len(tmp_sample_nodes_set):
                            frontiers = np.random.choice(frontiers.numpy(), num_nodes - len(tmp_sample_nodes_set), replace=False)
                            frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'average':
            for i in range(self.hop):
                tmp_sample_nodes = torch.cat(sample_nodes)
                tmp_sample_nodes = torch.unique(tmp_sample_nodes)
                num_nodes = int(self.max_num_nodes/self.hop*(i+1))
                if tmp_sample_nodes.size(0) < num_nodes:  # whether sample or not
                    frontiers = self.graph.out_edges(frontiers)[1]
                    frontiers = torch.unique(frontiers)
                    if frontiers.size(0) > num_nodes - tmp_sample_nodes.size(0):
                        frontiers = np.random.choice(frontiers.numpy(), num_nodes - tmp_sample_nodes.size(0),
                                                     replace=False)
                        frontiers = torch.unique(torch.tensor(frontiers))
                    sample_nodes.append(frontiers)
        elif self.sample_type == 'enclosing_subgraph':
            u, v = target_nodes[0], target_nodes[1]
            u_neighbour, v_neighbour = u, v
            # print(target_nodes)
            u_sample_nodes = [u_neighbour.reshape(1)]
            v_sample_nodes = [v_neighbour.reshape(1)]
            graph = self.graph
            if graph.has_edges_between(u, v):
                link_id = graph.edge_ids(u, v, return_uv=True)[2]
                graph.remove_edges(link_id)
            if graph.has_edges_between(v, u):
                link_id = graph.edge_ids(v, u, return_uv=True)[2]
                graph.remove_edges(link_id)
            for i in range(self.hop):
                u_frontiers = self.graph.out_edges(u_neighbour)[1]
                # v_frontiers = self.graph.out_edges(v_neighbour)[1]
                u_neighbour = u_frontiers
                # set(u_frontiers.tolist())
                if u_frontiers.size(0) > self.max_num_nodes:
                    u_frontiers = np.random.choice(u_frontiers.numpy(), self.max_num_nodes, replace=False)
                    u_frontiers = torch.tensor(u_frontiers)
                u_sample_nodes.append(u_frontiers)
            for i in range(self.hop):
                v_frontiers = self.graph.out_edges(v_neighbour)[1]
                # v_frontiers = self.graph.out_edges(v_neighbour)[1]
                v_neighbour = v_frontiers
                if v_frontiers.size(0) > self.max_num_nodes:
                    v_frontiers = np.random.choice(v_frontiers.numpy(), self.max_num_nodes, replace=False)
                    v_frontiers = torch.tensor(v_frontiers)
                v_sample_nodes.append(v_frontiers)
            # print('U', u_sample_nodes)
            # print('V', v_sample_nodes)
            u_sample_nodes = torch.cat(u_sample_nodes)
            u_sample_nodes = torch.unique(u_sample_nodes)
            v_sample_nodes = torch.cat(v_sample_nodes)
            v_sample_nodes = torch.unique(v_sample_nodes)
            # print('U', u_sample_nodes)
            # print('V', v_sample_nodes)
            u_sample_nodes_set = set(u_sample_nodes.tolist())
            v_sample_nodes_set = set(v_sample_nodes.tolist())
            uv_inter_neighbour = u_sample_nodes_set.intersection(v_sample_nodes_set)
            frontiers = torch.tensor(list(uv_inter_neighbour), dtype=torch.int64)
            # print(frontiers)
            sample_nodes.append(frontiers)
            # print(sample_nodes)
            # print("____________________________________")
            # exit(0)
                # v_neighbour = v_frontiers
        else:
            raise NotImplementedError

        sample_nodes = torch.cat(sample_nodes)
        sample_nodes = torch.unique(sample_nodes)
        # print(sample_nodes)
        # print("____________________________________")

        return sample_nodes

    def _collate(self, batch_nodes):

        # batch_graphs = map(list, zip(*batch))
        # print(batch_graphs)
        # print(batch_triples)
        # print(batch_labels)

        return batch_nodes

    def __call__(self, triples):
        sample_nodes_list = []
        edge_dataset = EdgeDataSet(triples, transform=self.sample_subgraph)
        self.print_fn('Using {} workers in sampling job.'.format(self.num_workers))
        sampler = DataLoader(edge_dataset, batch_size=1, num_workers=self.num_workers,
                             shuffle=False, collate_fn=self._collate)
        for sample_nodes in tqdm(sampler):
            sample_nodes_list.append(sample_nodes)
        # for sample_nodes in tqdm(sampler, ncols=100):
        #     print(sample_nodes)
        #     exit(0)
        #     sample_nodes_list.append(sample_nodes)

        return sample_nodes_list

class SEALData(object):
    """
    1. Generate positive and negative samples
    2. Subgraph sampling

    Attributes:
        g(dgl.DGLGraph): graph
        split_edge(dict): split edge
        hop(int): num of hop
        neg_samples(int): num of negative samples per positive sample
        subsample_ratio(float): ratio of subsample
        use_coalesce(bool): True for coalesce graph. Graph with multi-edge need to coalesce
    """

    def __init__(self, g, split_edge, hop=1, neg_samples=1, subsample_ratio=1, prefix=None, save_dir=None,
                 num_workers=32, shuffle=True, use_coalesce=True, print_fn=print):
        self.g = g
        self.hop = hop
        self.subsample_ratio = subsample_ratio
        self.prefix = prefix
        self.save_dir = save_dir
        self.print_fn = print_fn

        self.generator = PosNegEdgesGenerator(g=self.g,
                                              split_edge=split_edge,
                                              neg_samples=neg_samples,
                                              subsample_ratio=subsample_ratio,
                                              shuffle=shuffle)
        # if use_coalesce:
        #     for k, v in g.edata.items():
        #         g.edata[k] = v.float()  # dgl.to_simple() requires data is float
        #     self.g = dgl.to_simple(g, copy_ndata=True, copy_edata=True, aggregator='sum')
        #
        # self.ndata = {k: v for k, v in self.g.ndata.items()}
        # self.edata = {k: v for k, v in self.g.edata.items()}
        # self.g.ndata.clear()
        # self.g.edata.clear()
        # self.print_fn("Save ndata and edata in class.")
        # self.print_fn("Clear ndata and edata in graph.")
        #
        self.sampler = SEALSampler(graph=self.g,
                                   hop=hop,
                                   num_workers=num_workers,
                                   print_fn=print_fn)

    def __call__(self, split_type):

        if split_type == 'train':
            subsample_ratio = self.subsample_ratio
        else:
            subsample_ratio = 1

        path = osp.join(self.save_dir or '', '{}_{}_{}-hop_{}-subsample.bin'.format(self.prefix, split_type,
                                                                                    self.hop, subsample_ratio))

        if osp.exists(path):
            self.print_fn("Load existing processed {} files".format(split_type))
            graph_list, data = dgl.load_graphs(path)
            dataset = GraphDataSet(graph_list, data['labels'])

        else:
            self.print_fn("Processed {} files not exist.".format(split_type))

            edges, labels = self.generator(split_type)
            self.print_fn("Generate {} edges totally.".format(edges.size(0)))

            graph_list, labels = self.sampler(edges, labels)
            dataset = GraphDataSet(graph_list, labels)
            dgl.save_graphs(path, graph_list, {'labels': labels})
            self.print_fn("Save preprocessed subgraph to {}".format(path))
        return dataset


class TripleSampler(object):
    def __init__(self, g, data, sample_ratio=0.1):
        self.g = g
        self.sample_ratio = sample_ratio
        self.data = data

    def __call__(self, split_type):

        if split_type == 'train':
            sample_ratio = self.sample_ratio
            self.shuffle = True
            pos_edges = self.g.edges()
            pos_edges = torch.stack((pos_edges[0], pos_edges[1]), 1)

            g = add_self_loop(self.g)
            pos_edges = self.sample(pos_edges, sample_ratio).long()
            eids = g.edge_ids(pos_edges[:, 0], pos_edges[:, 1])
            edges = pos_edges
            # labels = torch.cat([torch.ones(pos_edges.size(0), 1), torch.zeros(neg_edges.size(0), 1)])
            triples = torch.stack(([edges[:, 0], g.edata['rel'][eids], edges[:, 1]]), 1)
        else:
            self.shuffle = False
            triples = torch.tensor(self.data[split_type])
            edges = torch.stack((triples[:, 0], triples[:, 2]), 1)

        if self.shuffle:
            perm = torch.randperm(edges.size(0))
            edges = edges[perm]
            triples = triples[perm]
        return edges, triples

    def sample(self, edges, sample_ratio):
        """
        Subsample generated edges.
        Args:
            edges(Tensor): edges to subsample
            subsample_ratio(float): ratio of subsample

        Returns:
            edges(Tensor):  edges

        """

        num_edges = edges.size(0)
        perm = torch.randperm(num_edges)
        perm = perm[:int(sample_ratio * num_edges)]
        edges = edges[perm]
        return edges


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

class GrailSampler(object):
    def __init__(self, dataset, file_paths, external_kg_file, db_path, hop, enclosing_sub_graph, max_nodes_per_hop):
        self.dataset = dataset
        self.file_paths = file_paths
        self.external_kg_file = external_kg_file
        self.db_path = db_path
        self.max_links = 2500000
        self.params = dict()
        self.params['hop'] = hop
        self.params['enclosing_sub_graph'] = enclosing_sub_graph
        self.params['max_nodes_per_hop'] = max_nodes_per_hop

    def generate_subgraph_datasets(self, num_neg_samples_per_link, constrained_neg_prob, splits=['train', 'valid', 'test'], saved_relation2id=None, max_label_value=None):


        testing = 'test' in splits
        #adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files(params.file_paths, saved_relation2id)

        # triple_file = f'data/{}/{}.txt'.format(params.dataset,params.BKG_file_name)
        if self.dataset == 'drugbank':
            adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = self.process_files_ddi(self.file_paths, self.external_kg_file, saved_relation2id)
        # else:
        #     adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel, triplets_mr, polarity_mr = self.process_files_decagon(params.file_paths, triple_file, saved_relation2id)
        # self.plot_rel_dist(adj_list, f'rel_dist.png')
        #print(triplets.keys(), triplets_mr.keys())
        data_path = f'datasets/{self.dataset}/relation2id.json'
        if not os.path.isdir(data_path) and testing:
            with open(data_path, 'w') as f:
                json.dump(relation2id, f)

        graphs = {}

        for split_name in splits:
            if self.dataset == 'drugbank':
                graphs[split_name] = {'triplets': triplets[split_name], 'max_size': self.max_links}
            # elif self.dataset == 'BioSNAP':
            #     graphs[split_name] = {'triplets': triplets_mr[split_name], 'max_size': params.max_links, "polarity_mr": polarity_mr[split_name]}
        # Sample train and valid/test links
        for split_name, split in graphs.items():
            print(f"Sampling negative links for {split_name}")
            split['pos'], split['neg'] = self.sample_neg(adj_list, split['triplets'], num_neg_samples_per_link, max_size=split['max_size'], constrained_neg_prob=constrained_neg_prob)
        #print(graphs.keys())
        # if testing:
        #     directory = os.path.join(params.main_dir, 'data/{}/'.format(params.dataset))
        #     save_to_file(directory, f'neg_{params.test_file}_{params.constrained_neg_prob}.txt', graphs['test']['neg'], id2entity, id2relation)

        self.links2subgraphs(adj_list, graphs, self.params, max_label_value)

    def process_files_ddi(self, files, triple_file, saved_relation2id=None, keeptrainone = False):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id

        triplets = {}
        kg_triple = []
        ent = 0
        rel = 0

        for file_type, file_path in files.items():
            data = []
            # with open(file_path) as f:
            #     file_data = [line.split() for line in f.read().split('\n')[:-1]]
            file_data = np.loadtxt(file_path)
            for triplet in file_data:
                #print(triplet)
                triplet[0], triplet[1], triplet[2] = int(triplet[0]), int(triplet[1]), int(triplet[2])
                if triplet[0] not in entity2id:
                    entity2id[triplet[0]] = triplet[0]
                    #ent += 1
                if triplet[1] not in entity2id:
                    entity2id[triplet[1]] = triplet[1]
                    #ent += 1
                if not saved_relation2id and triplet[2] not in relation2id:
                    if keeptrainone:
                        triplet[2] = 0
                        relation2id[triplet[2]] = 0
                        rel = 1
                    else:
                        relation2id[triplet[2]] = triplet[2]
                        rel += 1

                # Save the triplets corresponding to only the known relations
                if triplet[2] in relation2id:
                    data.append([entity2id[triplet[0]], entity2id[triplet[1]], relation2id[triplet[2]]])

            triplets[file_type] = np.array(data)
        #print(rel)
        triplet_kg = np.loadtxt(triple_file)
        # print(np.max(triplet_kg[:, -1]))
        for (h, t, r) in triplet_kg:
            h, t, r = int(h), int(t), int(r)
            if h not in entity2id:
                entity2id[h] = h
            if t not in entity2id:
                entity2id[t] = t
            if not saved_relation2id and rel+r not in relation2id:
                relation2id[rel+r] = rel + r
            kg_triple.append([h, t, r])
        kg_triple = np.array(kg_triple)
        id2entity = {v: k for k, v in entity2id.items()}
        id2relation = {v: k for k, v in relation2id.items()}
        #print(relation2id, rel)

        # Construct the list of adjacency matrix each corresponding to each relation. Note that this is constructed only from the train data.
        adj_list = []
        #print(kg_triple)
        #for i in range(len(relation2id)):
        for i in range(rel):
            idx = np.argwhere(triplets['train'][:, 2] == i)
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (triplets['train'][:, 0][idx].squeeze(1), triplets['train'][:, 1][idx].squeeze(1))), shape=(34124, 34124)))
        for i in range(rel, len(relation2id)):
            idx = np.argwhere(kg_triple[:, 2] == i-rel)
            #print(len(idx), i)
            adj_list.append(csc_matrix((np.ones(len(idx), dtype=np.uint8), (kg_triple[:, 0][idx].squeeze(1), kg_triple[:, 1][idx].squeeze(1))), shape=(34124, 34124)))
        #print(adj_list)
        #assert 0
        return adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel

    def plot_rel_dist(self, adj_list, filename):
        rel_count = []
        for adj in adj_list:
            rel_count.append(adj.count_nonzero())

        fig = plt.figure(figsize=(12, 8))
        plt.plot(rel_count)
        fig.savefig(filename, dpi=fig.dpi)

    def get_edge_count(self, adj_list):
        count = []
        for adj in adj_list:
            count.append(len(adj.tocoo().row.tolist()))
        return np.array(count)

    def sample_neg(self, adj_list, edges, num_neg_samples_per_link=1, max_size=1000000, constrained_neg_prob=0):
        pos_edges = edges
        neg_edges = []

        # if max_size is set, randomly sample train links
        if max_size < len(pos_edges):
            perm = np.random.permutation(len(pos_edges))[:max_size]
            pos_edges = pos_edges[perm]

        # sample negative links for train/test
        n, r = adj_list[0].shape[0], len(adj_list)

        # distribution of edges across reelations
        theta = 0.001
        edge_count = self.get_edge_count(adj_list)
        rel_dist = np.zeros(edge_count.shape)
        idx = np.nonzero(edge_count)
        rel_dist[idx] = softmax(theta * edge_count[idx])

        # possible head and tails for each relation
        valid_heads = [adj.tocoo().row.tolist() for adj in adj_list]
        valid_tails = [adj.tocoo().col.tolist() for adj in adj_list]

        pbar = tqdm(total=len(pos_edges))
        while len(neg_edges) < num_neg_samples_per_link * len(pos_edges):
            neg_head, neg_tail, rel = pos_edges[pbar.n % len(pos_edges)][0], pos_edges[pbar.n % len(pos_edges)][1], \
                pos_edges[pbar.n % len(pos_edges)][2]
            if np.random.uniform() < constrained_neg_prob:
                if np.random.uniform() < 0.5:
                    neg_head = np.random.choice(valid_heads[rel])
                else:
                    neg_tail = np.random.choice(valid_tails[rel])
            else:
                if np.random.uniform() < 0.5:
                    neg_head = np.random.choice(n)
                else:
                    neg_tail = np.random.choice(n)

            if neg_head != neg_tail and adj_list[rel][neg_head, neg_tail] == 0:
                neg_edges.append([neg_head, neg_tail, rel])
                pbar.update(1)

        pbar.close()

        neg_edges = np.array(neg_edges)
        return pos_edges, neg_edges

    def links2subgraphs(self, A, graphs, params, max_label_value=None):
        '''
        extract enclosing subgraphs, write map mode + named dbs
        '''
        max_n_label = {'value': np.array([0, 0])}
        subgraph_sizes = []
        enc_ratios = []
        num_pruned_nodes = []

        BYTES_PER_DATUM = self.get_average_subgraph_size(100, list(graphs.values())[0]['pos'], A, params) * 1.5
        links_length = 0
        for split_name, split in graphs.items():
            links_length += (len(split['pos']) + len(split['neg'])) * 2
        map_size = links_length * BYTES_PER_DATUM

        env = lmdb.open(self.db_path, map_size=map_size, max_dbs=6)

        def extraction_helper(A, links, g_labels, split_env):

            with env.begin(write=True, db=split_env) as txn:
                txn.put('num_graphs'.encode(), (len(links)).to_bytes(int.bit_length(len(links)), byteorder='little'))

            with mp.Pool(processes=None, initializer=self.intialize_worker, initargs=(A, params, max_label_value)) as p:
                args_ = zip(range(len(links)), links, g_labels)
                for (str_id, datum) in tqdm(p.imap(self.extract_save_subgraph, args_), total=len(links)):
                    max_n_label['value'] = np.maximum(np.max(datum['n_labels'], axis=0), max_n_label['value'])
                    subgraph_sizes.append(datum['subgraph_size'])
                    enc_ratios.append(datum['enc_ratio'])
                    num_pruned_nodes.append(datum['num_pruned_nodes'])

                    with env.begin(write=True, db=split_env) as txn:
                        txn.put(str_id, serialize(datum))

        for split_name, split in graphs.items():
            logging.info(f"Extracting enclosing subgraphs for positive links in {split_name} set")
            if self.dataset == 'BioSNAP':
                labels = np.array(split["polarity_mr"])
            else:
                labels = np.ones(len(split['pos']))
            db_name_pos = split_name + '_pos'
            split_env = env.open_db(db_name_pos.encode())
            extraction_helper(A, split['pos'], labels, split_env)

            logging.info(f"Extracting enclosing subgraphs for negative links in {split_name} set")
            if self.dataset == 'BioSNAP':
                labels = np.array(split["polarity_mr"])
            else:
                labels = np.ones(len(split['pos']))
            db_name_neg = split_name + '_neg'
            split_env = env.open_db(db_name_neg.encode())
            extraction_helper(A, split['neg'], labels, split_env)

        max_n_label['value'] = max_label_value if max_label_value is not None else max_n_label['value']

        with env.begin(write=True) as txn:
            bit_len_label_sub = int.bit_length(int(max_n_label['value'][0]))
            bit_len_label_obj = int.bit_length(int(max_n_label['value'][1]))
            txn.put('max_n_label_sub'.encode(),
                    (int(max_n_label['value'][0])).to_bytes(bit_len_label_sub, byteorder='little'))
            txn.put('max_n_label_obj'.encode(),
                    (int(max_n_label['value'][1])).to_bytes(bit_len_label_obj, byteorder='little'))

            txn.put('avg_subgraph_size'.encode(), struct.pack('f', float(np.mean(subgraph_sizes))))
            txn.put('min_subgraph_size'.encode(), struct.pack('f', float(np.min(subgraph_sizes))))
            txn.put('max_subgraph_size'.encode(), struct.pack('f', float(np.max(subgraph_sizes))))
            txn.put('std_subgraph_size'.encode(), struct.pack('f', float(np.std(subgraph_sizes))))

            txn.put('avg_enc_ratio'.encode(), struct.pack('f', float(np.mean(enc_ratios))))
            txn.put('min_enc_ratio'.encode(), struct.pack('f', float(np.min(enc_ratios))))
            txn.put('max_enc_ratio'.encode(), struct.pack('f', float(np.max(enc_ratios))))
            txn.put('std_enc_ratio'.encode(), struct.pack('f', float(np.std(enc_ratios))))

            txn.put('avg_num_pruned_nodes'.encode(), struct.pack('f', float(np.mean(num_pruned_nodes))))
            txn.put('min_num_pruned_nodes'.encode(), struct.pack('f', float(np.min(num_pruned_nodes))))
            txn.put('max_num_pruned_nodes'.encode(), struct.pack('f', float(np.max(num_pruned_nodes))))
            txn.put('std_num_pruned_nodes'.encode(), struct.pack('f', float(np.std(num_pruned_nodes))))

    def get_average_subgraph_size(self, sample_size, links, A, params):
        total_size = 0
        # print(links, len(links))
        lst = np.random.choice(len(links), sample_size)
        for idx in lst:
            (n1, n2, r_label) = links[idx]
            # for (n1, n2, r_label) in links[np.random.choice(len(links), sample_size)]:
            nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes, common_neighbor = self.subgraph_extraction_labeling((n1, n2),
                                                                                                       r_label, A,
                                                                                                       params['hop'],
                                                                                                       params['enclosing_sub_graph'],
                                                                                                       params['max_nodes_per_hop'])
            datum = {'nodes': nodes, 'r_label': r_label, 'g_label': 0, 'n_labels': n_labels, 'common_neighbor': common_neighbor,
                     'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
            total_size += len(serialize(datum))
        return total_size / sample_size

    def intialize_worker(self, A, params, max_label_value):
        global A_, params_, max_label_value_
        A_, params_, max_label_value_ = A, params, max_label_value

    def extract_save_subgraph(self, args_):
        idx, (n1, n2, r_label), g_label = args_
        nodes, n_labels, subgraph_size, enc_ratio, num_pruned_nodes, common_neighbor = self.subgraph_extraction_labeling((n1, n2), r_label,
                                                                                                   A_, params_['hop'],
                                                                                                   params_['enclosing_sub_graph'],
                                                                                                   params_['max_nodes_per_hop'])

        # max_label_value_ is to set the maximum possible value of node label while doing double-radius labelling.
        if max_label_value_ is not None:
            n_labels = np.array([np.minimum(label, max_label_value_).tolist() for label in n_labels])

        datum = {'nodes': nodes, 'r_label': r_label, 'g_label': g_label, 'n_labels': n_labels, 'common_neighbor': common_neighbor,
                 'subgraph_size': subgraph_size, 'enc_ratio': enc_ratio, 'num_pruned_nodes': num_pruned_nodes}
        str_id = '{:08}'.format(idx).encode('ascii')

        return (str_id, datum)

    def get_neighbor_nodes(self, roots, adj, h=1, max_nodes_per_hop=None):
        bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
        lvls = list()
        for _ in range(h):
            try:
                lvls.append(next(bfs_generator))
            except StopIteration:
                pass
        return set().union(*lvls)

    def subgraph_extraction_labeling(self, ind, rel, A_list, h=1, enclosing_sub_graph=False, max_nodes_per_hop=None,
                                     max_node_label_value=None):
        # extract the h-hop enclosing subgraphs around link 'ind'
        A_incidence = incidence_matrix(A_list)
        A_incidence += A_incidence.T
        ind = list(ind)
        ind[0], ind[1] = int(ind[0]), int(ind[1])
        ind = (ind[0], ind[1])
        root1_nei = self.get_neighbor_nodes(set([ind[0]]), A_incidence, h, max_nodes_per_hop)
        root2_nei = self.get_neighbor_nodes(set([ind[1]]), A_incidence, h, max_nodes_per_hop)
        subgraph_nei_nodes_int = root1_nei.intersection(root2_nei)
        subgraph_nei_nodes_un = root1_nei.union(root2_nei)

        root1_nei_1 = self.get_neighbor_nodes(set([ind[0]]), A_incidence, 1, max_nodes_per_hop)
        root2_nei_1 = self.get_neighbor_nodes(set([ind[1]]), A_incidence, 1, max_nodes_per_hop)
        common_neighbor = root1_nei_1.intersection(root2_nei_1)

        # Extract subgraph | Roots being in the front is essential for labelling and the model to work properly.
        if enclosing_sub_graph:
            if ind[0] in subgraph_nei_nodes_int:
                subgraph_nei_nodes_int.remove(ind[0])
            if ind[1] in subgraph_nei_nodes_int:
                subgraph_nei_nodes_int.remove(ind[1])
            subgraph_nodes = list(ind) + list(subgraph_nei_nodes_int)
        else:
            if ind[0] in subgraph_nei_nodes_un:
                subgraph_nei_nodes_un.remove(ind[0])
            if ind[1] in subgraph_nei_nodes_un:
                subgraph_nei_nodes_un.remove(ind[1])
            subgraph_nodes = list(ind) + list(subgraph_nei_nodes_un)  # list(set(ind).union(subgraph_nei_nodes_un))

        subgraph = [adj[subgraph_nodes, :][:, subgraph_nodes] for adj in A_list]

        labels, enclosing_subgraph_nodes = self.node_label(incidence_matrix(subgraph), max_distance=h)
        # print(ind, subgraph_nodes[:32],enclosing_subgraph_nodes[:32], labels)
        pruned_subgraph_nodes = np.array(subgraph_nodes)[enclosing_subgraph_nodes].tolist()
        pruned_labels = labels[enclosing_subgraph_nodes]
        # pruned_subgraph_nodes = subgraph_nodes
        # pruned_labels = labels

        if max_node_label_value is not None:
            pruned_labels = np.array([np.minimum(label, max_node_label_value).tolist() for label in pruned_labels])

        subgraph_size = len(pruned_subgraph_nodes)
        enc_ratio = len(subgraph_nei_nodes_int) / (len(subgraph_nei_nodes_un) + 1e-3)
        num_pruned_nodes = len(subgraph_nodes) - len(pruned_subgraph_nodes)
        # print(pruned_subgraph_nodes)
        # import time
        # time.sleep(10)
        return pruned_subgraph_nodes, pruned_labels, subgraph_size, enc_ratio, num_pruned_nodes, common_neighbor

    def node_label(self, subgraph, max_distance=1):
        # implementation of the node labeling scheme described in the paper
        roots = [0, 1]
        sgs_single_root = [remove_nodes(subgraph, [root]) for root in roots]
        dist_to_roots = [
            np.clip(ssp.csgraph.dijkstra(sg, indices=[0], directed=False, unweighted=True, limit=1e6)[:, 1:], 0, 1e7)
            for r, sg in enumerate(sgs_single_root)]
        dist_to_roots = np.array(list(zip(dist_to_roots[0][0], dist_to_roots[1][0])), dtype=int)

        target_node_labels = np.array([[0, 1], [1, 0]])
        labels = np.concatenate((target_node_labels, dist_to_roots)) if dist_to_roots.size else target_node_labels

        enclosing_subgraph_nodes = np.where(np.max(labels, axis=1) <= max_distance)[0]
        return labels, enclosing_subgraph_nodes