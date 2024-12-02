import numpy as np
import scipy.sparse as ssp
import random
from scipy.sparse import csc_matrix

"""All functions in this file are from  dgl.contrib.data.knowledge_graph"""


def _bfs_relational(adj, roots, max_nodes_per_hop=None):
    """
    BFS for graphs.
    Modified from dgl.contrib.data.knowledge_graph to accomodate node sampling
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        if max_nodes_per_hop and max_nodes_per_hop < len(next_lvl):
            next_lvl = set(random.sample(next_lvl, max_nodes_per_hop))

        yield next_lvl

        current_lvl = set.union(next_lvl)


def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors.
    Directly copied from dgl.contrib.data.knowledge_graph"""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(ssp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return ssp.csr_matrix((data, (row_ind, col_ind)), shape=shape)


def process_files_ddi(files, triple_file, saved_relation2id=None, keeptrainone = False):
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


def get_neighbor_nodes(roots, adj, h=1, max_nodes_per_hop=None):
    bfs_generator = _bfs_relational(adj, roots, max_nodes_per_hop)
    lvls = list()
    for _ in range(h):
        try:
            lvls.append(next(bfs_generator))
        except StopIteration:
            pass
    return set().union(*lvls)