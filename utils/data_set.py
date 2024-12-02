from torch.utils.data import Dataset
import numpy as np
import torch
import dgl
from dgl import NID
from scipy.sparse.csgraph import shortest_path
import lmdb
import pickle
from utils.dgl_utils import get_neighbor_nodes


class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, num_rel, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent
        self.num_rel = num_rel

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            triple, label, pos_neg = torch.tensor(ele['triple'], dtype=torch.long), torch.tensor(ele['label'], dtype=torch.long), torch.tensor(ele['pos_neg'], dtype=torch.float)
            return triple, label, pos_neg
        else:
            triple, label, random_hop = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label']), torch.tensor(ele['random_hop'])
        label = triple[1]
        # label = torch.tensor(label, dtype=torch.long)
        # label = self.get_label_rel(label)
        # if self.label_smooth != 0.0:
        #     label = (1.0 - self.label_smooth) * label + (1.0 / self.num_rel)
        if self.p.search_mode == 'random':
            return triple, label, random_hop
        else:
            return triple, label

    def get_label_rel(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, num_rel, params):
        super(TestDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_ent = num_ent
        self.num_rel = num_rel

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            triple, label, pos_neg = torch.tensor(ele['triple'], dtype=torch.long), torch.tensor(ele['label'], dtype=torch.long), torch.tensor(ele['pos_neg'], dtype=torch.float)
            return triple, label, pos_neg
        else:
            triple, label, random_hop = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label']), torch.tensor(ele['random_hop'])
        label = triple[1]
        # label = torch.tensor(label, dtype=torch.long)
        # label = self.get_label_rel(label)
        if self.p.search_mode == 'random':
            return triple, label, random_hop
        else:
            return triple, label

    def get_label_rel(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)


class GraphTrainDataset(Dataset):
    def __init__(self, triplets, num_ent, num_rel, params, all_graph, db_name_pos=None):
        super(GraphTrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.g = all_graph
        db_path = f'subgraph/{self.p.dataset}/{self.p.subgraph_type}_neg_{self.p.num_neg_samples_per_link}_hop_{self.p.subgraph_hop}_seed_{self.p.seed}'
        if self.p.save_mode == 'mdb':
            self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
            self.db_pos = self.main_env.open_db(db_name_pos.encode())
            # with self.main_env.begin(db=self.db_pos) as txn:
            #     num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
            #     print(num_graphs_pos)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        if self.p.save_mode == 'pickle':
            sample_nodes, triple, label = ele['sample_nodes'], torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
            subgraph = self.subgraph_sample(triple[0], triple[2],  sample_nodes)
        elif self.p.save_mode == 'graph':
            subgraph, triple, label = ele['subgraph'], torch.tensor(ele['triple'], dtype=torch.long), torch.tensor(ele['triple'][1], dtype=torch.long)
        elif self.p.save_mode == 'mdb':
            # triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
            with self.main_env.begin(db=self.db_pos) as txn:
                str_id = '{:08}'.format(item).encode('ascii')
                nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
                # print(nodes_pos)
                # head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels_pos])
                # tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels_pos])
                # print(head_id)
                # print(nodes_pos[head_id[0]], nodes_pos[tail_id[0]])
                # exit(0)
                # n_ids = np.zeros(len(nodes_pos))
                # n_ids[head_id] = 1  # head
                # n_ids[tail_id] = 2  # tail
                # subgraph.ndata['id'] = torch.FloatTensor(n_ids)
                if self.p.train_mode == 'tune':
                    subgraph = torch.zeros(self.num_ent, dtype=torch.bool)
                    subgraph[nodes_pos] = 1
                else:
                    subgraph = self.subgraph_sample(nodes_pos[0], nodes_pos[1], nodes_pos)
                triple = torch.tensor([nodes_pos[0], r_label_pos,nodes_pos[1]], dtype=torch.long)
                return subgraph, triple, torch.tensor(r_label_pos, dtype=torch.long)
        # input_ids = self.convert_subgraph_to_tokens(subgraph, self.max_seq_length)
        # label = self.get_label_rel(label)
        # if self.label_smooth != 0.0:
        #     label = (1.0 - self.label_smooth) * label + (1.0 / self.num_rel)
        return subgraph, triple, label

    def get_label_rel(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

    def subgraph_sample(self, u, v, sample_nodes):
        subgraph = dgl.node_subgraph(self.g, sample_nodes)
        n_ids = np.zeros(len(sample_nodes))
        n_ids[0] = 1  # head
        n_ids[1] = 2  # tail
        subgraph.ndata['id'] = torch.tensor(n_ids, dtype=torch.long)
        # print(sample_nodes)
        # print(u,v)
        # Each node should have unique node id in the new subgraph
        u_id = int(torch.nonzero(subgraph.ndata[NID] == int(u), as_tuple=False))
        # print(torch.nonzero(subgraph.ndata[NID] == int(u), as_tuple=False))
        # print(torch.nonzero(subgraph.ndata[NID] == int(v), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[NID] == int(v), as_tuple=False))

        if subgraph.has_edges_between(u_id, v_id):
            link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)
        if subgraph.has_edges_between(v_id, u_id):
            link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)

        n_ids = np.zeros(len(sample_nodes))
        n_ids[0] = 1  # head
        n_ids[1] = 2  # tail
        subgraph.ndata['id'] = torch.tensor(n_ids, dtype=torch.long)

        # z = drnl_node_labeling(subgraph, u_id, v_id)
        # subgraph.ndata['z'] = z
        return subgraph


class GraphTestDataset(Dataset):
    def __init__(self, triplets, num_ent, num_rel, params, all_graph, db_name_pos=None):
        super(GraphTestDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.g = all_graph
        db_path = f'subgraph/{self.p.dataset}/{self.p.subgraph_type}_neg_{self.p.num_neg_samples_per_link}_hop_{self.p.subgraph_hop}_seed_{self.p.seed}'
        if self.p.save_mode == 'mdb':
            self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
            self.db_pos = self.main_env.open_db(db_name_pos.encode())

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        if self.p.save_mode == 'pickle':
            sample_nodes, triple, label = ele['sample_nodes'], torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
            subgraph = self.subgraph_sample(triple[0], triple[2],  sample_nodes)
        elif self.p.save_mode == 'graph':
            subgraph, triple, label = ele['subgraph'], torch.tensor(ele['triple'], dtype=torch.long), torch.tensor(ele['triple'][1], dtype=torch.long)
        elif self.p.save_mode == 'mdb':
            # triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
            with self.main_env.begin(db=self.db_pos) as txn:
                str_id = '{:08}'.format(item).encode('ascii')
                nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()
                if self.p.train_mode == 'tune':
                    subgraph = torch.zeros(self.num_ent, dtype=torch.bool)
                    subgraph[nodes_pos] = 1
                else:
                    subgraph = self.subgraph_sample(nodes_pos[0], nodes_pos[1], nodes_pos)
                triple = torch.tensor([nodes_pos[0], r_label_pos, nodes_pos[1]], dtype=torch.long)
                return subgraph, triple, torch.tensor(r_label_pos, dtype=torch.long)
        # label = self.get_label_rel(label)
        return subgraph, triple, label

    def get_label_rel(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_rel*2], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

    def subgraph_sample(self, u, v, sample_nodes):
        subgraph = dgl.node_subgraph(self.g, sample_nodes)
        n_ids = np.zeros(len(sample_nodes))
        n_ids[0] = 1  # head
        n_ids[1] = 2  # tail
        subgraph.ndata['id'] = torch.tensor(n_ids, dtype=torch.long)
        # Each node should have unique node id in the new subgraph
        u_id = int(torch.nonzero(subgraph.ndata[NID] == int(u), as_tuple=False))
        v_id = int(torch.nonzero(subgraph.ndata[NID] == int(v), as_tuple=False))

        if subgraph.has_edges_between(u_id, v_id):
            link_id = subgraph.edge_ids(u_id, v_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)
        if subgraph.has_edges_between(v_id, u_id):
            link_id = subgraph.edge_ids(v_id, u_id, return_uv=True)[2]
            subgraph.remove_edges(link_id)

        # z = drnl_node_labeling(subgraph, u_id, v_id)
        # subgraph.ndata['z'] = z
        return subgraph


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

def deserialize(data):
    data_tuple = pickle.loads(data)
    keys = ('nodes', 'r_label',  'g_label', 'n_label', 'common_neighbor')
    return dict(zip(keys, data_tuple))

class NCNDataset(Dataset):
    def __init__(self, triplets, num_ent, num_rel, params, adj, db_name_pos=None):
        super(NCNDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.adj = adj
        db_path = f'subgraph/{self.p.dataset}/{self.p.subgraph_type}_neg_{self.p.num_neg_samples_per_link}_hop_{self.p.subgraph_hop}_seed_{self.p.seed}'
        if self.p.save_mode == 'mdb':
            self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
            self.db_pos = self.main_env.open_db(db_name_pos.encode())
            # with self.main_env.begin(db=self.db_pos) as txn:
            #     num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
            #     print(num_graphs_pos)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        with self.main_env.begin(db=self.db_pos) as txn:
            str_id = '{:08}'.format(item).encode('ascii')
            nodes_pos, r_label_pos, g, n, cn = deserialize(txn.get(str_id)).values()
            triple = torch.tensor([nodes_pos[0], r_label_pos,nodes_pos[1]], dtype=torch.long)
            label = torch.tensor(r_label_pos, dtype=torch.long)
            cn_index = torch.zeros([self.num_ent+1], dtype=torch.bool)
            if len(cn) == 0:
                cn_index[self.num_ent] = 1
            else:
                cn_index[list(cn)] = 1
            return triple, label, cn_index

    def get_common_neighbors(self, u, v):
        cns_list = []
        # for i_u in range(1, self.p.n_layer+1):
        #     for i_v in range(1, self.p.n_layer+1):
        #         root_u_nei = get_neighbor_nodes({u}, self.adj, i_u)
        #         root_v_nei = get_neighbor_nodes({v}, self.adj, i_v)
        #         subgraph_nei_nodes_int = root_u_nei.intersection(root_v_nei)
        #         ng = list(subgraph_nei_nodes_int)
        #         subgraph = torch.zeros([1, self.num_ent], dtype=torch.bool)
        #         if len(ng) == 0:
        #             cns_list.append(subgraph)
        #             continue
        #         subgraph[:,ng] = 1
        #         cns_list.append(subgraph)
        # root_u_nei = get_neighbor_nodes({u}, self.adj, 1)
        # root_v_nei = get_neighbor_nodes({v}, self.adj, 1)
        # subgraph_nei_nodes_int = root_u_nei.intersection(root_v_nei)
        # ng = list(subgraph_nei_nodes_int)
        # subgraph = torch.zeros([1, self.num_ent], dtype=torch.bool)
        # if len(ng) == 0:
        #     return subgraph
        # else:
        #     subgraph[:,ng] = 1
        return 1