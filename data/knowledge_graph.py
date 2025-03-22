"""
based on the implementation in DGL
(https://github.com/dmlc/dgl/blob/master/python/dgl/contrib/data/knowledge_graph.py)
Knowledge graph dataset for Relational-GCN
Code adapted from authors' implementation of Relational-GCN
https://github.com/tkipf/relational-gcn
https://github.com/MichSchli/RelationPrediction
"""

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import scipy.sparse as sp
import os

from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url
from utils.dgl_utils import process_files_ddi
from utils.graph_utils import incidence_matrix

# np.random.seed(123)

_downlaod_prefix = _get_dgl_url('dataset/')


def load_data(dataset):
    if dataset in ['drugbank', 'twosides', 'twosides_200', 'drugbank_s1', 'twosides_s1']:
        return load_link(dataset)
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))


class RGCNLinkDataset(object):

    def __init__(self, name):
        self.name = name
        self.dir = 'datasets'
       
        # zip_path = os.path.join(self.dir, '{}.zip'.format(self.name))
        self.dir = os.path.join(self.dir, self.name)
        # extract_archive(zip_path, self.dir)

    def load(self):
        entity_path = os.path.join(self.dir, 'entities.dict')
        relation_path = os.path.join(self.dir, 'relations.dict')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = np.asarray(_read_triplets_as_list(
            train_path, entity_dict, relation_dict))
        self.valid = np.asarray(_read_triplets_as_list(
            valid_path, entity_dict, relation_dict))
        self.test = np.asarray(_read_triplets_as_list(
            test_path, entity_dict, relation_dict))
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = len(relation_dict)
        print("# relations: {}".format(self.num_rels))
        print("# training sample: {}".format(len(self.train)))
        print("# valid sample: {}".format(len(self.valid)))
        print("# testing sample: {}".format(len(self.test)))
        file_paths = {
            'train': f'{self.dir}/train_raw.txt',
            'valid': f'{self.dir}/dev_raw.txt',
            'test': f'{self.dir}/test_raw.txt'
        }
        # external_kg_file = f'{self.dir}/external_kg.txt'
        # adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(file_paths, external_kg_file)
        # A_incidence = incidence_matrix(adj_list)
        # A_incidence += A_incidence.T
        # self.adj = A_incidence



def load_link(dataset):
    if 'twosides' in dataset or 'ogbl_biokg' in dataset:
        data = MultiLabelDataset(dataset)
    else:
        data = RGCNLinkDataset(dataset)
    data.load()
    return data


def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d


def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


def _read_multi_rel_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_multi_rel_triplets_as_array(filename, entity_dict):
    graph_list = []
    input_list = []
    multi_label_list = []
    pos_neg_list = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        o = entity_dict[triplet[1]]
        r_list = list(map(int, triplet[2].split(',')))
        multi_label_list.append(r_list)
        r_label = [i for i, _ in enumerate(r_list) if _ == 1]
        for r in r_label:
            graph_list.append([s, r, o])
        input_list.append([s, -1, o])
        pos_neg = int(triplet[3])
        pos_neg_list.append(pos_neg)
    return np.asarray(graph_list), np.asarray(input_list), np.asarray(multi_label_list), np.asarray(pos_neg_list)

class MultiLabelDataset(object):
    def __init__(self, name):
        self.name = name
        self.dir = 'datasets'

        # zip_path = os.path.join(self.dir, '{}.zip'.format(self.name))
        self.dir = os.path.join(self.dir, self.name)
        # extract_archive(zip_path, self.dir)

    def load(self):
        entity_path = os.path.join(self.dir, 'entities.dict')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        self.train_graph, self.train_input, self.train_multi_label, self.train_pos_neg = _read_multi_rel_triplets_as_array(
            train_path, entity_dict)
        _, self.valid_input, self.valid_multi_label, self.valid_pos_neg = _read_multi_rel_triplets_as_array(
            valid_path, entity_dict)
        _, self.test_input, self.test_multi_label, self.test_pos_neg = _read_multi_rel_triplets_as_array(
            test_path, entity_dict)
        self.num_nodes = len(entity_dict)
        print("# entities: {}".format(self.num_nodes))
        self.num_rels = self.train_multi_label.shape[1]
        print("# relations: {}".format(self.num_rels))
        print("# training sample: {}".format(self.train_input.shape[0]))
        print("# valid sample: {}".format(self.valid_input.shape[0]))
        print("# testing sample: {}".format(self.test_input.shape[0]))
        # print("# training sample: {}".format(len(self.train)))
        # print("# valid sample: {}".format(len(self.valid)))
        # print("# testing sample: {}".format(len(self.test)))
        # file_paths = {
        #     'train': f'{self.dir}/train_raw.txt',
        #     'valid': f'{self.dir}/dev_raw.txt',
        #     'test': f'{self.dir}/test_raw.txt'
        # }
        # external_kg_file = f'{self.dir}/external_kg.txt'
        # adj_list, triplets, entity2id, relation2id, id2entity, id2relation, rel = process_files_ddi(file_paths,
        #                                                                                             external_kg_file)
        # A_incidence = incidence_matrix(adj_list)
        # A_incidence += A_incidence.T
        # self.adj = A_incidence