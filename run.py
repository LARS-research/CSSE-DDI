import csv
import os
import argparse
import time
import logging
from pprint import pprint
import numpy as np
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
import dgl
from data.knowledge_graph import load_data
from model import GCN_TransE, GCN_DistMult, GCN_ConvE, SubgraphSelector, GCN_ConvE_Rel, GCN_Transformer, GCN_None, \
    GCN_MLP, GCN_MLP_NCN, SearchGCN_MLP, SearchedGCN_MLP, NetworkGNN_MLP, SearchGCN_MLP_SPOS, SEAL_GCN
from model.lte_models import TransE, DistMult, ConvE
from utils import process, process_multi_label, TrainDataset, TestDataset, get_logger, GraphTrainDataset, GraphTestDataset, \
    get_f1_score_list, get_acc_list, get_neighbor_nodes, NCNDataset, Temp_Scheduler
import wandb
from os.path import exists
from os import mkdir, makedirs
from dgl.dataloading import GraphDataLoader
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import setproctitle
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK, rand, space_eval
from model.genotypes import *
import optuna
from optuna.samplers import RandomSampler
from utils import CategoricalASNG
import itertools
from sortedcontainers import SortedDict
from dgl import NID, EID

torch.multiprocessing.set_sharing_strategy('file_system')


class Runner(object):
    def __init__(self, params):
        self.p = params
        self.prj_path = os.getcwd()
        self.data = load_data(self.p.dataset)
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.num_ent, self.train_data, self.num_rels = self.data.num_nodes, self.data.train_graph, self.data.num_rels
            # self.train_input, self.valid_rel, self.test_rel = self.data.train_rel, self.data.valid_rel, self.data.test_rel
            # self.train_pos_neg, self.valid_pos_neg, self.test_pos_neg = self.data.train_pos_neg, self.data.valid_pos_neg, self.data.test_pos_neg
            self.triplets = process_multi_label(
                {'train': self.data.train_input, 'valid': self.data.valid_input, 'test': self.data.test_input},
                {'train': self.data.train_multi_label, 'valid': self.data.valid_multi_label, 'test': self.data.test_multi_label},
                {'train': self.data.train_pos_neg, 'valid': self.data.valid_pos_neg, 'test': self.data.test_pos_neg}
            )
        else:
            self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, self.data.train, self.data.valid, self.data.test, self.data.num_rels
            self.triplets, self.class2num = process(
                {'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                self.num_rels, self.p.n_layer, self.p.add_reverse)
        self.p.embed_dim = self.p.k_w * \
                           self.p.k_h if self.p.embed_dim is None else self.p.embed_dim  # output dim of gnn
        self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()
        if self.p.input_type == "subgraph":
            self.get_subgraph()
        self.data_iter = self.get_data_iter()

        if (self.p.search_mode != 'arch_random' or self.p.search_mode != 'arch_search') and self.p.search_algorithm!='random_ps2':
            self.model = self.get_model()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.best_val_f1, self.best_val_auroc, self.best_epoch, self.best_val_results, self.best_test_results = 0., 0., 0., {}, {}
        self.best_test_f1, self.best_test_auroc = 0., 0.
        self.early_stop_cnt = 0
        os.makedirs(f'logs/{self.p.dataset}/', exist_ok=True)
        if self.p.train_mode == 'tune':
            tmp_name = self.p.name + '_tune'
            self.logger = get_logger(f'logs/{self.p.dataset}/', tmp_name)
        else:
            self.logger = get_logger(f'logs/{self.p.dataset}/', self.p.name)
        pprint(vars(self.p))

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def save_search_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent + 1)
        g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
        if self.p.add_reverse:
            g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        return g

    def get_data_iter(self):

        def get_data_loader(dataset_class, split, shuffle=True):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.num_rels, self.p),
                batch_size=self.p.batch_size,
                shuffle=shuffle,
                num_workers=self.p.num_workers
            )

        def get_graph_data_loader(dataset_class, split, db_name_pos=None):
            return GraphDataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.num_rels, self.p, self.g, db_name_pos),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        def get_ncndata_loader(dataset_class, split, db_name_pos=None):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.num_rels, self.p, self.adj, db_name_pos),
                batch_size=self.p.batch_size,
                shuffle=True,
                num_workers=self.p.num_workers
            )

        if self.p.input_type == 'subgraph' or self.p.fine_tune_with_implicit_subgraph:
            if self.p.add_reverse:
                return {
                    'train_rel': get_graph_data_loader(GraphTrainDataset, 'train_rel'),
                    'valid_rel': get_graph_data_loader(GraphTestDataset, 'valid_rel'),
                    'valid_rel_inv': get_graph_data_loader(GraphTestDataset, 'valid_rel_inv'),
                    'test_rel': get_graph_data_loader(GraphTestDataset, 'test_rel'),
                    'test_rel_inv': get_graph_data_loader(GraphTestDataset, 'test_rel_inv'),
                    # 'valid_head': get_data_loader(TestDataset, 'valid_head'),
                    # 'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
                    # 'test_head': get_data_loader(TestDataset, 'test_head'),
                    # 'test_tail': get_data_loader(TestDataset, 'test_tail'),
                }
            else:
                return {
                    'train_rel': get_graph_data_loader(GraphTrainDataset, 'train_rel', 'train_pos'),
                    'valid_rel': get_graph_data_loader(GraphTestDataset, 'valid_rel', 'valid_pos'),
                    'test_rel': get_graph_data_loader(GraphTestDataset, 'test_rel', 'test_pos'),
                }
        elif self.p.input_type == 'allgraph' and self.p.score_func == 'mlp_ncn':
            return {
                'train_rel': get_ncndata_loader(NCNDataset, 'train_rel', 'train_pos'),
                'valid_rel': get_ncndata_loader(NCNDataset, 'valid_rel', 'valid_pos'),
                'test_rel': get_ncndata_loader(NCNDataset, 'test_rel', 'test_pos'),
            }
        else:
            if self.p.add_reverse:
                return {
                    'train_rel': get_data_loader(TrainDataset, 'train_rel'),
                    'valid_rel': get_data_loader(TestDataset, 'valid_rel'),
                    'valid_rel_inv': get_data_loader(TestDataset, 'valid_rel_inv'),
                    'test_rel': get_data_loader(TestDataset, 'test_rel'),
                    'test_rel_inv': get_data_loader(TestDataset, 'test_rel_inv')
                }
            else:
                return {
                    'train_rel': get_data_loader(TrainDataset, 'train_rel'),
                    'valid_rel': get_data_loader(TestDataset, 'valid_rel'),
                    'test_rel': get_data_loader(TestDataset, 'test_rel')
                }

    def get_edge_dir_and_norm(self):
        """
        :return: edge_type: indicates type of each edge: [E]
        """
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float()
        norm = in_deg ** -0.5
        norm[torch.isinf(norm).bool()] = 0
        self.g.ndata['xxx'] = norm
        self.g.apply_edges(
            lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})
        if self.p.gpu >= 0:
            norm = self.g.edata.pop('xxx').squeeze().to("cuda:0")
            if self.p.add_reverse:
                edge_type = torch.tensor(np.concatenate(
                    [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to("cuda:0")
            else:
                edge_type = torch.tensor(self.train_data[:, 1]).to("cuda:0")
        else:
            norm = self.g.edata.pop('xxx').squeeze()
            edge_type = torch.tensor(np.concatenate(
                [self.train_data[:, 1], self.train_data[:, 1] + self.num_rels]))
        return edge_type, norm

    def get_model(self):
        if self.p.n_layer > 0:
            if self.p.score_func.lower() == 'transe':
                model = GCN_TransE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                   init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                   n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                   bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                   hid_drop=self.p.hid_drop, gamma=self.p.gamma, wni=self.p.wni, wsi=self.p.wsi,
                                   encoder=self.p.encoder, use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.encoder == 'gcn':
                model = SEAL_GCN(self.num_ent, self.num_rels, self.p.init_dim, self.p.gcn_dim, self.p.embed_dim, self.p.n_layer, loss_type=self.p.loss_type)
            elif self.p.score_func.lower() == 'distmult':
                model = GCN_DistMult(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                     init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                     n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                     bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                     hid_drop=self.p.hid_drop, wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
                                     use_bn=(not self.p.nobn), ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'conve':
                model = GCN_ConvE(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                  init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                  n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                  bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                  hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                  conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                  num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                  wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn),
                                  ltr=(not self.p.noltr))
            elif self.p.score_func.lower() == 'conve_rel':
                model = GCN_ConvE_Rel(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                      init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                      n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                      bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                      hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                      conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                      num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                      wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder, use_bn=(not self.p.nobn),
                                      ltr=(not self.p.noltr), input_type=self.p.input_type)
            elif self.p.score_func.lower() == 'transformer':
                model = GCN_Transformer(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                        init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                        n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                        bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                        hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                        conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                        num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                        wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
                                        use_bn=(not self.p.nobn),
                                        ltr=(not self.p.noltr), input_type=self.p.input_type,
                                        d_model=self.p.d_model, num_transformer_layers=self.p.num_transformer_layers,
                                        nhead=self.p.nhead, dim_feedforward=self.p.dim_feedforward,
                                        transformer_dropout=self.p.transformer_dropout,
                                        transformer_activation=self.p.transformer_activation,
                                        graph_pooling=self.p.graph_pooling_type, concat_type=self.p.concat_type,
                                        max_input_len=self.p.subgraph_max_num_nodes, loss_type=self.p.loss_type)
            elif self.p.score_func.lower() == 'none':
                model = GCN_None(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                 init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                 n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                 bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                 hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                 conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                 num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                 wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
                                 use_bn=(not self.p.nobn),
                                 ltr=(not self.p.noltr), input_type=self.p.input_type,
                                 graph_pooling=self.p.graph_pooling_type, concat_type=self.p.concat_type,
                                 loss_type=self.p.loss_type, add_reverse=self.p.add_reverse)
            elif self.p.score_func.lower() == 'mlp' and self.p.encoder != 'searchgcn' and self.p.genotype is None:
                model = GCN_MLP(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
                                use_bn=(not self.p.nobn),
                                ltr=(not self.p.noltr), input_type=self.p.input_type,
                                graph_pooling=self.p.graph_pooling_type, combine_type=self.p.combine_type,
                                loss_type=self.p.loss_type, add_reverse=self.p.add_reverse)
            elif self.p.score_func.lower() == 'mlp_ncn':
                model = GCN_MLP_NCN(num_ent=self.num_ent, num_rel=self.num_rels, num_base=self.p.num_bases,
                                    init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                    n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                    bias=self.p.bias, gcn_drop=self.p.gcn_drop, opn=self.p.opn,
                                    hid_drop=self.p.hid_drop, input_drop=self.p.input_drop,
                                    conve_hid_drop=self.p.conve_hid_drop, feat_drop=self.p.feat_drop,
                                    num_filt=self.p.num_filt, ker_sz=self.p.ker_sz, k_h=self.p.k_h, k_w=self.p.k_w,
                                    wni=self.p.wni, wsi=self.p.wsi, encoder=self.p.encoder,
                                    use_bn=(not self.p.nobn),
                                    ltr=(not self.p.noltr), input_type=self.p.input_type,
                                    graph_pooling=self.p.graph_pooling_type, combine_type=self.p.combine_type,
                                    loss_type=self.p.loss_type, add_reverse=self.p.add_reverse)
            elif self.p.genotype is not None:
                model = SearchedGCN_MLP(args=self.p, num_ent=self.num_ent, num_rel=self.num_rels,
                                        num_base=self.p.num_bases,
                                        init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                        n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                        bias=self.p.bias, gcn_drop=self.p.gcn_drop, hid_drop=self.p.hid_drop,
                                        input_drop=self.p.input_drop,
                                        wni=self.p.wni, wsi=self.p.wsi, use_bn=(not self.p.nobn),
                                        ltr=(not self.p.noltr),
                                        combine_type=self.p.combine_type, loss_type=self.p.loss_type,
                                        genotype=self.p.genotype)
            elif "spos" in self.p.search_algorithm or self.p.train_mode=='vis_hop' or (self.p.train_mode=='spos_tune' and self.p.weight_sharing == True):
                model = SearchGCN_MLP_SPOS(args=self.p, num_ent=self.num_ent, num_rel=self.num_rels,
                                      num_base=self.p.num_bases,
                                      init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                      n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                      bias=self.p.bias, gcn_drop=self.p.gcn_drop, hid_drop=self.p.hid_drop,
                                      input_drop=self.p.input_drop,
                                      wni=self.p.wni, wsi=self.p.wsi, use_bn=(not self.p.nobn), ltr=(not self.p.noltr),
                                      combine_type=self.p.combine_type, loss_type=self.p.loss_type)
            elif self.p.score_func.lower() == 'mlp' and self.p.genotype is None:
                model = SearchGCN_MLP(args=self.p, num_ent=self.num_ent, num_rel=self.num_rels,
                                      num_base=self.p.num_bases,
                                      init_dim=self.p.init_dim, gcn_dim=self.p.gcn_dim, embed_dim=self.p.embed_dim,
                                      n_layer=self.p.n_layer, edge_type=self.edge_type, edge_norm=self.edge_norm,
                                      bias=self.p.bias, gcn_drop=self.p.gcn_drop, hid_drop=self.p.hid_drop,
                                      input_drop=self.p.input_drop,
                                      wni=self.p.wni, wsi=self.p.wsi, use_bn=(not self.p.nobn), ltr=(not self.p.noltr),
                                      combine_type=self.p.combine_type, loss_type=self.p.loss_type)
            else:
                raise KeyError(
                    f'score function {self.p.score_func} not recognized.')
        else:
            if self.p.score_func.lower() == 'transe':
                model = TransE(self.num_ent, self.num_rels, params=self.p)
            elif self.p.score_func.lower() == 'distmult':
                model = DistMult(self.num_ent, self.num_rels, params=self.p)
            elif self.p.score_func.lower() == 'conve':
                model = ConvE(self.num_ent, self.num_rels, params=self.p)
            else:
                raise NotImplementedError

        if self.p.gpu >= 0:
            model.to("cuda:0")
        return model

    def get_subgraph(self):
        subgraph_dir = f'subgraph/{args.dataset}/{self.p.subgraph_type}_{self.p.subgraph_hop}_{self.p.subgraph_max_num_nodes}_{self.p.subgraph_sample_type}_{self.p.seed}'
        if not exists(subgraph_dir):
            makedirs(subgraph_dir)

        for mode in ['train_rel', 'valid_rel', 'test_rel']:
            if self.p.subgraph_is_saved:
                if self.p.save_mode == 'pickle':
                    with open(
                            f'{subgraph_dir}/{mode}_{self.p.subgraph_type}_{self.p.subgraph_hop}_{self.p.subgraph_max_num_nodes}_{self.p.subgraph_sample_type}_{self.p.seed}.pkl',
                            'rb') as f:
                        sample_nodes = pickle.load(f)
                    # graph_list = dgl.load_graphs(f'{subgraph_dir}/{mode}_{self.p.subgraph_type}_{self.p.subgraph_hop}_{self.p.subgraph_max_num_nodes}_{self.p.subgraph_sample_type}_{self.p.seed}.bin')[0]
                    for idx, _ in enumerate(self.triplets[mode]):
                        self.triplets[mode][idx]['sample_nodes'] = sample_nodes[idx][0]
                elif self.p.save_mode == 'graph':
                    if self.p.add_reverse:
                        graph_list = dgl.load_graphs(
                            f'{subgraph_dir}/{mode}_add_reverse.bin')[0]
                    else:
                        graph_list = dgl.load_graphs(
                            f'{subgraph_dir}/{mode}.bin')[0]
                    for idx, _ in enumerate(self.triplets[mode]):
                        self.triplets[mode][idx]['subgraph'] = graph_list[idx]
    def random_search(self):
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        os.makedirs(save_root, exist_ok=True)
        save_path = f'{save_root}/{self.p.name}_random.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch_rs()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='random')
            if self.p.dataset == 'drugbank':
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if self.p.dataset == 'drugbank':
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                wandb.log({"train_loss": train_loss, "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 15:
                self.logger.info("Early stop!")
                break
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='random')
        end = time.time()
        if self.p.dataset == 'drugbank':
            self.logger.info(
                f"f1: Rel {test_results['left_f1']:.5}, Rel_rev {test_results['right_f1']:.5}, Avg {test_results['macro_f1']:.5}")
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "test_acc": test_results['acc'],
                "test_f1": test_results['macro_f1'],
                "test_cohen": test_results['kappa']
            })

    def train(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.train_mode}/{self.p.name}/'
        os.makedirs(save_root, exist_ok=True)
        save_path = f'{save_root}/model_weight.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='normal')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_auroc": self.best_val_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        # self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='normal')
        end = time.time()
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            # wandb.log({
            #     "test_auroc": test_results['auroc'],
            #     "test_auprc": test_results['auprc'],
            #     "test_ap": test_results['ap']
            # })
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            # wandb.log({
            #     "test_acc": test_results['acc'],
            #     "test_f1": test_results['macro_f1'],
            #     "test_cohen": test_results['kappa']
            # })

    def train_epoch(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model(g, subj, obj, cns)
            else:
                if self.p.encoder == 'gcn':
                    pred = self.model(g, g.ndata['z'])
                else:
                    pred = self.model(g, subj, obj)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                loss = self.model.calc_loss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            if self.p.clip_grad:
                clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    def evaluate_epoch(self, split, mode='normal'):

        def get_combined_results(left, right):
            results = dict()
            results['acc'] = round((left['acc'] + right['acc']) / 2, 5)
            results['left_f1'] = round(left['macro_f1'], 5)
            results['right_f1'] = round(right['macro_f1'], 5)
            results['macro_f1'] = round((left['macro_f1'] + right['macro_f1']) / 2, 5)
            results['kappa'] = round((left['kappa'] + right['kappa']) / 2, 5)
            results['macro_f1_per_class'] = (np.array(left['macro_f1_per_class']) + np.array(
                right['macro_f1_per_class'])) / 2.0
            results['acc_per_class'] = (np.array(left['acc_per_class']) + np.array(right['acc_per_class'])) / 2.0
            return results

        def get_results(left):
            results = dict()
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                results['auroc'] = round(left['auroc'], 5)
                results['auprc'] = round(left['auprc'], 5)
                results['ap'] = round(left['ap'], 5)
            else:
                results['acc'] = round(left['acc'], 5)
                # results['auc_pr'] = round((left['auc_pr'] + right['auc_pr']) / 2, 5)
                # results['micro_f1'] = round((left['micro_f1'] + right['micro_f1']) / 2, 5)
                results['macro_f1'] = round(left['macro_f1'], 5)
                results['kappa'] = round(left['kappa'], 5)
            return results

        self.model.eval()
        if mode == 'normal':
            if self.p.add_reverse:
                left_result, left_loss = self.predict(split, '')
                right_result, right_loss = self.predict(split, '_inv')
            else:
                left_result, left_loss = self.predict(split, '')
        elif mode == 'normal_mix_hop':
            left_result, left_loss = self.predict_mix_hop(split, '')
        elif mode == 'random':
            left_result, left_loss = self.predict_rs(split, '')
            right_result, right_loss = self.predict_rs(split, '_inv')
        elif mode == 'ps2':
            if self.p.add_reverse:
                left_result, left_loss = self.predict_search(split, '')
                right_result, right_loss = self.predict_search(split, '_inv')
            else:
                left_result, left_loss = self.predict_search(split, '')
        elif mode == 'arch_search':
            left_result, left_loss = self.predict_arch_search(split, '')
        elif mode == 'arch_search_s':
            left_result, left_loss = self.predict_arch_search(split, '', 'evaluate_single_path')
        elif mode == 'joint_search':
            left_result, left_loss = self.predict_joint_search(split, '')
        elif mode == 'joint_search_s':
            left_result, left_loss = self.predict_joint_search(split, '', 'evaluate_single_path')
        elif mode == 'spos_train_supernet':
            left_result, left_loss = self.predict_spos_search(split, '')
        elif mode == 'spos_arch_search':
            left_result, left_loss = self.predict_spos_search(split, '', spos_mode='arch_search')
        elif mode == 'spos_train_supernet_ps2':
            left_result, left_loss = self.predict_spos_search_ps2(split, '')
        elif mode == 'spos_arch_search_ps2':
            left_result, left_loss = self.predict_spos_search_ps2(split, '', spos_mode='arch_search')
        # res = get_results(left_result)
        # return res, left_loss
        if self.p.add_reverse:
            res = get_combined_results(left_result, right_result)
            return res, (left_loss + right_loss) / 2.0
        else:
            return left_result, left_loss

    def predict(self, split, mode):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        with torch.no_grad():
            results = dict()
            eval_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(eval_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                if self.p.score_func == 'mlp_ncn':
                    cns = batch[2].to("cuda:0")
                    pred = self.model(g, subj, obj, cns)
                else:
                    if self.p.encoder == 'gcn':
                        pred = self.model(g, g.ndata['z'])
                    else:
                        pred = self.model(g, subj, obj)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            # dict_res = metrics.classification_report(pos_labels, pos_scores, output_dict=True, zero_division=1)
            # results['macro_f1_per_class'] = get_f1_score_list(dict_res)
            # results['acc_per_class'] = get_acc_list(dict_res)
            # self.logger.info(f'Macro f1 per class: {results["macro_f1_per_class"]}')
            # self.logger.info(f'Acc per class: {results["acc_per_class"]}')
            loss = np.mean(loss_list)
        return results, loss

    def train_epoch_rs(self):
        self.model.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels, input_ids = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0"), \
                    batch[3].to("cuda:0")
            else:
                triplets, labels, random_hops = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to('cuda:0')
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_rel = self.model.forward_search(g, subj, obj)
            pred = self.model.compute_pred_rs(hidden_all_ent, all_rel, subj, obj, random_hops)
            # pred = self.model(g, subj, obj, random_hops)  # [batch_size, num_ent]
            loss = self.model.calc_loss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        loss = np.mean(loss_list)
        return loss

    def predict_rs(self, split, mode):
        loss_list = []
        pos_scores = []
        pos_labels = []
        self.model.eval()
        with torch.no_grad():
            results = dict()
            eval_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(eval_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels, input_ids = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to(
                        "cuda:0"), batch[3].to("cuda:0")
                else:
                    triplets, labels, random_hops = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to('cuda:0')
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_rel = self.model.forward_search(g, subj, obj)
                pred = self.model.compute_pred_rs(hidden_all_ent, all_rel, subj, obj, random_hops)
                loss = self.model.calc_loss(pred, labels)
                loss_list.append(loss.item())
                pos_labels += rel.to('cpu').numpy().flatten().tolist()
                pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
            results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
            results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
            results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            loss = np.mean(loss_list)
        return results, loss

    def fine_tune(self):
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        save_model_path = f'{save_root}/{self.p.name}.pt'
        save_ss_path = f'{save_root}/{self.p.name}_ss.pt'
        save_tune_path = f'{save_root}/{self.p.name}_tune.pt'
        self.model.load_state_dict(torch.load(str(save_model_path)))
        self.subgraph_selector.load_state_dict(torch.load(str(save_ss_path)))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)
        val_results, val_loss = self.evaluate_epoch('valid', 'ps2')
        test_results, test_loss = self.evaluate_epoch('test', 'ps2')
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Validation]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            wandb.log({
                "init_test_auroc": test_results['auroc'],
                "init_test_auprc": test_results['auprc'],
                "init_test_ap": test_results['ap']
            })
        else:
            self.logger.info(
                f"[Validation]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "init_test_acc": test_results['acc'],
                "init_test_f1": test_results['macro_f1'],
                "init_test_cohen": test_results['kappa']
            })
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch_fine_tune()
            val_results, valid_loss = self.evaluate_epoch('valid', 'ps2')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_tune_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_tune_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_auroc": self.best_val_auroc})
                self.scheduler.step(self.best_val_auroc)
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_f1": self.best_val_f1})
                self.scheduler.step(self.best_val_f1)
            if self.early_stop_cnt == 15:
                self.logger.info("Early stop!")
                break
        self.load_model(save_tune_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results, test_loss = self.evaluate_epoch('test', 'ps2')
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            wandb.log({
                "test_auroc": test_results['auroc'],
                "test_auprc": test_results['auprc'],
                "test_ap": test_results['ap']
            })
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "test_acc": test_results['acc'],
                "test_f1": test_results['macro_f1'],
                "test_cohen": test_results['kappa']
            })

    def train_epoch_fine_tune(self, mode=None):
        self.model.train()
        self.subgraph_selector.eval()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=mode)  # [batch_size, num_ent]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector,
                                               mode='argmax')
            else:
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, mode='argmax', search_algorithm=self.p.ss_search_algorithm)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                train_loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                train_loss = self.model.calc_loss(pred, labels)
            loss_list.append(train_loss.item())
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

        loss = np.mean(loss_list)
        return loss

    def ps2(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        os.makedirs(save_root, exist_ok=True)
        self.logger = get_logger(f'{save_root}/',f'train')
        save_model_path = f'{save_root}/weight.pt'
        save_ss_path = f'{save_root}/weight_ss.pt'

        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        self.subgraph_selector_optimizer = torch.optim.Adam(
            self.subgraph_selector.parameters(), lr=self.p.ss_lr, weight_decay=self.p.l2)
        # temp_scheduler = Temp_Scheduler(self.p.max_epochs, self.p.temperature, self.p.temperature, temp_min=self.p.temperature_min)
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            # if self.p.cos_temp:
            #     self.p.temperature = temp_scheduler.step()
            # else:
            #     self.p.temperature = self.p.temperature
            train_loss = self.search_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', 'ps2')
            test_results, test_loss = self.evaluate_epoch('test', 'ps2')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_test_results = test_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_test_auroc = test_results['auroc']
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), str(save_model_path))
                    torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_test_results = test_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_test_f1 = test_results['macro_f1']
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), str(save_model_path))
                    torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_auroc": self.best_val_auroc})
                wandb.log({'best_test_auroc': self.best_test_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_f1": self.best_val_f1})
                # wandb.log({'best_test_f1': self.best_test_f1})
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            # self.logger.info(
            #     f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            wandb.log({
                "test_auroc": self.best_test_results['auroc'],
                "test_auprc": self.best_test_results['auprc'],
                "test_ap": self.best_test_results['ap']
            })
        else:
            # self.logger.info(
            #     f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "test_acc": self.best_test_results['acc'],
                "test_f1": self.best_test_results['macro_f1'],
                "test_cohen": self.best_test_results['kappa']
            })


    def search_epoch(self):
        self.model.train()
        self.subgraph_selector.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        valid_iter = self.data_iter['valid_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector)
            else:
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, search_algorithm=self.p.ss_search_algorithm)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                train_loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                train_loss = self.model.calc_loss(pred, labels)
            loss_list.append(train_loss.item())
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.subgraph_selector_optimizer.zero_grad()

            batch = next(iter(valid_iter))
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector)
            else:
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, search_algorithm=self.p.ss_search_algorithm)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                valid_loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                valid_loss = self.model.calc_loss(pred, labels)
            valid_loss.backward()
            self.subgraph_selector_optimizer.step()
        loss = np.mean(loss_list)
        return loss

    def predict_search(self, split, mode):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        self.subgraph_selector.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(test_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
                if self.p.score_func == 'mlp_ncn':
                    cns = batch[2].to("cuda:0")
                    pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector,
                                                   mode='argmax')
                else:
                    pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, mode='argmax', search_algorithm=self.p.ss_search_algorithm)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            # dict_res = metrics.classification_report(pos_labels, pos_scores, output_dict=True, zero_division=1)
            # results['macro_f1_per_class'] = get_f1_score_list(dict_res)
            # results['acc_per_class'] = get_acc_list(dict_res)
            loss = np.mean(loss_list)
        return results, loss

    def architecture_search(self):
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        os.makedirs(save_root, exist_ok=True)
        save_model_path = f'{save_root}/{self.p.name}.pt'

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.p.max_epochs, eta_min=self.p.lr_min)
        self.arch_optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=self.p.arch_lr, weight_decay=self.p.arch_weight_decay)
        self.arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.arch_optimizer, self.p.max_epochs, eta_min=self.p.arch_lr_min)
        temp_scheduler = Temp_Scheduler(self.p.max_epochs, self.p.temperature, self.p.temperature, temp_min=self.p.temperature_min)
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            # genotype = self.model.genotype()
            # self.logger.info(f'Genotype: {genotype}')
            if self.p.cos_temp:
                self.p.temperature = temp_scheduler.step()
            else:
                self.p.temperature = self.p.temperature
            # print(self.p.temperature)
            train_loss = self.arch_search_epoch()
            self.scheduler.step()
            self.arch_scheduler.step()
            val_results, valid_loss = self.evaluate_epoch('valid', 'arch_search')
            s_val_results, s_valid_loss = self.evaluate_epoch('valid', 'arch_search_s')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    # self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    # self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            genotype = self.model.genotype()
            self.logger.info(f'[Epoch {epoch}]: LR: {self.scheduler.get_last_lr()[0]}, TEMP: {self.p.temperature}, Genotype: {genotype}')
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Valid_S Loss: {s_valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Valid_S AUROC: {s_val_results['auroc']:.5}, Valid_S AUPRC: {s_val_results['auprc']:.5}, Valid_S AP@50: {s_val_results['ap']:.5}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Valid_S ACC: {s_val_results['acc']:.5}, Valid_S Macro F1: {s_val_results['macro_f1']:.5}, Valid_S Cohen: {s_val_results['kappa']:.5}")
            # if self.early_stop_cnt == 15:
            #     self.logger.info("Early stop!")
            #     break

    def arch_search_epoch(self):
        self.model.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        valid_iter = self.data_iter['valid_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            for update_idx in range(self.p.w_update_epoch):
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                # hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
                pred = self.model(g, subj, obj)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    loss = self.model.calc_loss(pred, labels, pos_neg)
                else:
                    loss = self.model.calc_loss(pred, labels)
                self.arch_optimizer.zero_grad()
                loss.backward(retain_graph=True)
                if self.p.clip_grad:
                    clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                loss_list.append(loss.item())
            if self.p.alpha_mode == 'train_loss':
                self.arch_optimizer.step()
            elif self.p.alpha_mode == 'valid_loss':
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                batch = next(iter(valid_iter))
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred = self.model(g, subj, obj)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    valid_loss = self.model.calc_loss(pred, labels, pos_neg)
                else:
                    valid_loss = self.model.calc_loss(pred, labels)
                valid_loss.backward()
                self.arch_optimizer.step()
        loss = np.mean(loss_list)
        return loss

    def predict_arch_search(self, split, mode, eval_mode=None):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(test_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                # hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
                pred = self.model(g, subj, obj, mode=eval_mode)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            # dict_res = metrics.classification_report(pos_labels, pos_scores, output_dict=True, zero_division=1)
            # results['macro_f1_per_class'] = get_f1_score_list(dict_res)
            # results['acc_per_class'] = get_acc_list(dict_res)
            loss = np.mean(loss_list)
        return results, loss

    def arch_random_search_each(self, trial):
        genotype_space = []
        for i in range(self.p.n_layer):
            genotype_space.append(trial.suggest_categorical("mess"+ str(i), COMP_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("agg"+ str(i), AGG_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("comb"+ str(i), COMB_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("act"+ str(i), ACT_PRIMITIVES))
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        # self.best_valid_metric, self.best_test_metric = 0.0, {}
        self.p.genotype = "||".join(genotype_space)
        # run = self.reinit_wandb()
        self.model = self.get_model().to("cuda:0")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        os.makedirs(save_root, exist_ok=True)
        save_path = f'{save_root}/random_search.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='normal')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                    # self.logger.info("Update best valid auroc!")
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                    # self.logger.info("Update best valid f1!")
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_auroc": self.best_val_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        # self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        self.logger.info(f'{self.p.genotype}')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='normal')
        end = time.time()
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            # wandb.log({
            #     "test_auroc": test_results['auroc'],
            #     "test_auprc": test_results['auprc'],
            #     "test_ap": test_results['ap']
            # })
            # run.finish()
            if self.best_val_auroc > self.best_valid_metric:
                self.best_valid_metric = self.best_val_auroc
                self.best_test_metric = test_results
            with open(f'{save_root}/random_search_arch_list.csv', "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.p.genotype, self.best_val_auroc, test_results['auroc']])
            return self.best_val_auroc
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            # wandb.log({
            #     "test_acc": test_results['acc'],
            #     "test_f1": test_results['macro_f1'],
            #     "test_cohen": test_results['kappa']
            # })
            # run.finish()
            if self.best_val_f1 > self.best_valid_metric:
                self.best_valid_metric = self.best_val_f1
                self.best_test_metric = test_results
            with open(f'{save_root}/random_search_arch_list.csv', "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.p.genotype, self.best_val_f1, test_results['macro_f1']])
            return self.best_val_f1

    def arch_random_search(self):
        self.best_valid_metric = 0.0
        self.best_test_metric = {}
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
        os.makedirs(save_root, exist_ok=True)
        self.logger = get_logger(f'{save_root}/', f'random_search')
        study = optuna.create_study(directions=["maximize"], sampler=RandomSampler())
        study.optimize(self.arch_random_search_each, n_trials=self.p.baseline_sample_num)
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
        with open(f'{save_root}/random_search_res.txt', "w") as f1:
            f1.write(f'{self.p.__dict__}\n')
            f1.write(f'{self.p.genotype}\n')
            f1.write(f'Valid performance: {study.best_value}\n')
            f1.write(f'Test performance: {self.best_test_metric}')

    def train_parameter(self, parameter):
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        self.p.genotype = "||".join(parameter)
        run = self.reinit_wandb()
        self.model = self.get_model().to("cuda:0")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        os.makedirs(save_root, exist_ok=True)
        save_path = f'{save_root}/{self.p.name}.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='normal')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                    self.logger.info("Update best valid auroc!")
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                    self.logger.info("Update best valid f1!")
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_auroc": self.best_val_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 15:
                self.logger.info("Early stop!")
                break
        # self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='normal')
        end = time.time()
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            wandb.log({
                "test_auroc": test_results['auroc'],
                "test_auprc": test_results['auprc'],
                "test_ap": test_results['ap']
            })
        else:
            if self.p.add_reverse:
                self.logger.info(
                    f"f1: Rel {test_results['left_f1']:.5}, Rel_rev {test_results['right_f1']:.5}, Avg {test_results['macro_f1']:.5}")
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "test_acc": test_results['acc'],
                "test_f1": test_results['macro_f1'],
                "test_cohen": test_results['kappa']
            })
        run.finish()
        return {'loss': -self.best_val_f1, "status": STATUS_OK}

    def reinit_wandb(self):
        if self.p.train_mode == 'spos_tune':
            run = wandb.init(
                reinit=True,
                project=self.p.wandb_project,
                settings=wandb.Settings(start_method="fork"),
                config={
                    "dataset": self.p.dataset,
                    "encoder": self.p.encoder,
                    "score_function": self.p.score_func,
                    "batch_size": self.p.batch_size,
                    "learning_rate": self.p.lr,
                    "weight_decay":self.p.l2,
                    "encoder_layer_num": self.p.n_layer,
                    "epochs": self.p.max_epochs,
                    "seed": self.p.seed,
                    "train_mode": self.p.train_mode,
                    "init_dim": self.p.init_dim,
                    "embed_dim": self.p.embed_dim,
                    "input_type": self.p.input_type,
                    "loss_type": self.p.loss_type,
                    "search_mode": self.p.search_mode,
                    "combine_type": self.p.combine_type,
                    "genotype": self.p.genotype,
                    "exp_note": self.p.exp_note,
                    "alpha_mode": self.p.alpha_mode,
                    "few_shot_op": self.p.few_shot_op,
                    "tune_sample_num": self.p.tune_sample_num
                }
            )
        elif self.p.search_mode == 'arch_random':
            run = wandb.init(
                reinit=True,
                project=self.p.wandb_project,
                settings=wandb.Settings(start_method="fork"),
                config={
                    "dataset": self.p.dataset,
                    "encoder": self.p.encoder,
                    "score_function": self.p.score_func,
                    "batch_size": self.p.batch_size,
                    "learning_rate": self.p.lr,
                    "weight_decay":self.p.l2,
                    "encoder_layer_num": self.p.n_layer,
                    "epochs": self.p.max_epochs,
                    "seed": self.p.seed,
                    "train_mode": self.p.train_mode,
                    "init_dim": self.p.init_dim,
                    "embed_dim": self.p.embed_dim,
                    "input_type": self.p.input_type,
                    "loss_type": self.p.loss_type,
                    "search_mode": self.p.search_mode,
                    "combine_type": self.p.combine_type,
                    "genotype": self.p.genotype,
                    "exp_note": self.p.exp_note,
                    "tune_sample_num": self.p.tune_sample_num
                }
            )
        return run

    def joint_search(self):
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        os.makedirs(save_root, exist_ok=True)
        save_model_path = f'{save_root}/{self.p.name}.pt'
        save_ss_path = f'{save_root}/{self.p.name}_ss.pt'
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        self.upper_optimizer = torch.optim.Adam([{'params': self.model.arch_parameters()},
                                                 {'params': self.subgraph_selector.parameters(), 'lr': self.p.ss_lr}],
                                                lr=self.p.arch_lr, weight_decay=self.p.arch_weight_decay)
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            # genotype = self.model.genotype()
            # self.logger.info(f'Genotype: {genotype}')
            if self.p.cos_temp and epoch % 5 == 0 and epoch != 0:
                self.p.temperature = self.p.temperature * 0.5
            else:
                self.p.temperature = self.p.temperature
            train_loss = self.joint_search_epoch()
            # self.scheduler.step()
            # self.arch_scheduler.step()
            val_results, valid_loss = self.evaluate_epoch('valid', 'joint_search')
            s_val_results, s_valid_loss = self.evaluate_epoch('valid', 'joint_search_s')
            if val_results['macro_f1'] > self.best_val_f1:
                self.best_val_results = val_results
                self.best_val_f1 = val_results['macro_f1']
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), str(save_model_path))
                torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
                self.early_stop_cnt = 0
            else:
                self.early_stop_cnt += 1
            genotype = self.model.genotype()
            # self.logger.info(f'[Epoch {epoch}]: LR: {self.scheduler.get_last_lr()[0]}, TEMP: {self.p.temperature}, Genotype: {genotype}')
            self.logger.info(
                f'[Epoch {epoch}]: TEMP: {self.p.temperature}, Genotype: {genotype}')
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Valid_S Loss: {s_valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if self.p.dataset == 'drugbank':
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Valid_S ACC: {s_val_results['acc']:.5}, Valid Macro F1: {s_val_results['macro_f1']:.5}, Valid Cohen: {s_val_results['kappa']:.5}")
            if self.early_stop_cnt == 50:
                self.logger.info("Early stop!")
                break

    def joint_search_epoch(self):
        self.model.train()
        self.subgraph_selector.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        valid_iter = self.data_iter['valid_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector)
            else:
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, search_algorithm=self.p.search_algorithm)
            train_loss = self.model.calc_loss(pred, labels)
            loss_list.append(train_loss.item())
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.upper_optimizer.zero_grad()

            batch = next(iter(valid_iter))
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            if self.p.score_func == 'mlp_ncn':
                cns = batch[2].to("cuda:0")
                pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector)
            else:
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, search_algorithm=self.p.search_algorithm)
            valid_loss = self.model.calc_loss(pred, labels)
            valid_loss.backward()
            self.upper_optimizer.step()
        loss = np.mean(loss_list)
        return loss

    def predict_joint_search(self, split, mode, eval_mode=None):
        loss_list = []
        pos_scores = []
        pos_labels = []
        self.model.eval()
        self.subgraph_selector.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(test_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_ent = self.model.forward_search(g, mode=eval_mode)  # [batch_size, num_ent]
                if self.p.score_func == 'mlp_ncn':
                    cns = batch[2].to("cuda:0")
                    pred = self.model.compute_pred(hidden_all_ent, all_ent, subj, obj, cns, self.subgraph_selector,
                                                   mode='argmax')
                else:
                    pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, mode='argmax', search_algorithm=self.p.search_algorithm)
                eval_loss = self.model.calc_loss(pred, labels)
                loss_list.append(eval_loss.item())
                pos_labels += rel.to('cpu').numpy().flatten().tolist()
                pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
            results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
            results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
            results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            loss = np.mean(loss_list)
        return results, loss

    def joint_tune(self):
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}'
        save_model_path = f'{save_root}/{self.p.name}.pt'
        save_ss_path = f'{save_root}/{self.p.name}_ss.pt'
        save_tune_path = f'{save_root}/{self.p.name}_tune.pt'
        self.model.load_state_dict(torch.load(str(save_model_path)))
        self.subgraph_selector.load_state_dict(torch.load(str(save_ss_path)))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)

        val_results, val_loss = self.evaluate_epoch('valid', 'joint_search_s')
        test_results, test_loss = self.evaluate_epoch('test', 'joint_search_s')
        if self.p.dataset == 'drugbank':
            # if self.p.add_reverse:
            #     self.logger.info(
            #         f"f1: Rel {test_results['left_f1']:.5}, Rel_rev {test_results['right_f1']:.5}, Avg {test_results['macro_f1']:.5}")
            self.logger.info(
                f"[Validation]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "init_test_acc": test_results['acc'],
                "init_test_f1": test_results['macro_f1'],
                "init_test_cohen": test_results['kappa']
            })
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch_fine_tune(mode='evaluate_single_path')
            val_results, valid_loss = self.evaluate_epoch('valid', 'joint_search_s')
            if val_results['macro_f1'] > self.best_val_f1:
                self.best_val_results = val_results
                self.best_val_f1 = val_results['macro_f1']
                self.best_epoch = epoch
                self.save_model(save_tune_path)
                self.early_stop_cnt = 0
            else:
                self.early_stop_cnt += 1
                # torch.save(self.model.state_dict(), str(save_model_path))
                # torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
            genotype = self.model.genotype()
            self.logger.info(
                f'[Epoch {epoch}]: Genotype: {genotype}')
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            self.logger.info(
                f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
            self.logger.info(
                f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
            wandb.log({"train_loss": train_loss, "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 15:
                self.logger.info("Early stop!")
                break
            self.scheduler.step(self.best_val_f1)
        self.load_model(save_tune_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results, test_loss = self.evaluate_epoch('test', 'joint_search_s')
        test_acc = test_results['acc']
        test_f1 = test_results['macro_f1']
        test_cohen = test_results['kappa']
        wandb.log({
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_cohen": test_cohen
        })
        if self.p.dataset == 'drugbank':
            test_acc = test_results['acc']
            test_f1 = test_results['macro_f1']
            test_cohen = test_results['kappa']
            wandb.log({
                "test_acc": test_acc,
                "test_f1": test_f1,
                "test_cohen": test_cohen
            })
            if self.p.add_reverse:
                self.logger.info(
                    f"f1: Rel {test_results['left_f1']:.5}, Rel_rev {test_results['right_f1']:.5}, Avg {test_results['macro_f1']:.5}")
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")

    def spos_train_supernet(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        print(save_root)
        os.makedirs(save_root, exist_ok=True)
        if self.p.weight_sharing:
            self.logger = get_logger(f'{save_root}/', f'train_supernet_ws_{self.p.few_shot_op}')
            save_model_path = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{"_".join(save_root.split("/")[-1].split("_")[:-3])}/400.pt'
            self.model.load_state_dict(torch.load(save_model_path))
        else:
            self.logger = get_logger(f'{save_root}/', f'train_supernet')
        for epoch in range(1, self.p.max_epochs+1):
            start_time = time.time()
            train_loss = self.architecture_search_spos_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', 'spos_train_supernet')
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
            # self.scheduler.step(train_loss)
            wandb.log({
                "train_loss": train_loss,
                "valid_loss": valid_loss
            })
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'{save_root}/{epoch}.pt')

    def architecture_search_spos_epoch(self):
        self.model.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            self.generate_single_path()
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            # hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            pred = self.model(g, subj, obj)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                loss = self.model.calc_loss(pred, labels)
            loss.backward()
            if self.p.clip_grad:
                clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            loss_list.append(loss.item())
        loss = np.mean(loss_list)
        return loss

    def predict_spos_search(self, split, mode, eval_mode=None, spos_mode='train_supernet'):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(test_iter):
                if spos_mode == 'train_supernet':
                    self.generate_single_path()
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                # hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
                pred = self.model(g, subj, obj, mode=eval_mode)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            loss = np.mean(loss_list)
        return results, loss

    def spos_arch_search(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        os.makedirs(save_root, exist_ok=True)
        for save_epoch in [800,700,600,500,400,300,200,100]:
            try:
                self.model.load_state_dict(torch.load(f'{save_root}/{save_epoch}.pt'))
            except:
                continue
            self.logger = get_logger(f'{save_root}/', f'{save_epoch}_arch_search')
            valid_loss_searched_arch_res = dict()
            valid_f1_searched_arch_res = dict()
            valid_auroc_searched_arch_res = dict()
            search_time = 0.0
            t_start = time.time()
            for epoch in range(1, self.p.spos_arch_sample_num + 1):
                self.generate_single_path()
                arch = "||".join(self.model.ops)
                val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search')
                test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search')
                valid_loss_searched_arch_res.setdefault(arch, valid_loss)
                self.logger.info(f'[Epoch {epoch}]: Path:{arch}')
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    self.logger.info(
                        f"[Epoch {epoch}]: Valid Loss: {valid_loss:.5}, Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                    self.logger.info(
                        f"[Epoch {epoch}]: Test Loss: {test_loss:.5}, Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
                    valid_auroc_searched_arch_res.setdefault(arch, val_results['auroc'])
                else:
                    self.logger.info(
                        f"[Epoch {epoch}]: Valid Loss: {valid_loss:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid ACC: {val_results['acc']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                    self.logger.info(
                        f"[Epoch {epoch}]: Test Loss: {test_loss:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test ACC: {test_results['acc']:.5}, Test Cohen: {test_results['kappa']:.5}")
                    valid_f1_searched_arch_res.setdefault(arch, val_results['macro_f1'])

            t_end = time.time()
            search_time = (t_end - t_start)

            search_time = search_time / 3600
            self.logger.info(f'The search process costs {search_time:.2f}h.')
            import csv
            with open(f'{save_root}/valid_loss_{save_epoch}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['arch', 'valid loss'])
                valid_loss_searched_arch_res_sorted = sorted(valid_loss_searched_arch_res.items(), key=lambda x :x[1])
                res = valid_loss_searched_arch_res_sorted
                for i in range(len(res)):
                    writer.writerow([res[i][0], res[i][1]])
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                with open(f'{save_root}/valid_auroc_{save_epoch}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid auroc'])
                    valid_auroc_searched_arch_res_sorted = sorted(valid_auroc_searched_arch_res.items(), key=lambda x: x[1],
                                                               reverse=True)
                    res = valid_auroc_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])
            else:
                with open(f'{save_root}/valid_f1_{save_epoch}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid f1'])
                    valid_f1_searched_arch_res_sorted = sorted(valid_f1_searched_arch_res.items(), key=lambda x: x[1], reverse=True)
                    res = valid_f1_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])

    def generate_single_path(self):
        if self.p.exp_note is None:
            self.model.ops = self.model.generate_single_path()
        elif self.p.exp_note == 'only_search_act':
            self.model.ops = self.model.generate_single_path_act()
        elif self.p.exp_note == 'only_search_comb':
            self.model.ops = self.model.generate_single_path_comb()
        elif self.p.exp_note == 'only_search_comp':
            self.model.ops = self.model.generate_single_path_comp()
        elif self.p.exp_note == 'only_search_agg':
            self.model.ops = self.model.generate_single_path_agg()
        elif self.p.exp_note == 'only_search_agg_comb':
            self.model.ops = self.model.generate_single_path_agg_comb()
        elif self.p.exp_note == 'only_search_agg_comb_comp':
            self.model.ops = self.model.generate_single_path_agg_comb_comp()
        elif self.p.exp_note == 'only_search_agg_comb_act_rotate':
            self.model.ops = self.model.generate_single_path_agg_comb_act_rotate()
        elif self.p.exp_note == 'only_search_agg_comb_act_mult':
            self.model.ops = self.model.generate_single_path_agg_comb_act_mult()
        elif self.p.exp_note == 'only_search_agg_comb_act_ccorr':
            self.model.ops = self.model.generate_single_path_agg_comb_act_ccorr()
        elif self.p.exp_note == 'only_search_agg_comb_act_sub':
            self.model.ops = self.model.generate_single_path_agg_comb_act_sub()
        elif self.p.exp_note == 'spfs' and self.p.search_algorithm == 'spos_arch_search_ps2':
            self.model.ops = self.model.generate_single_path()
        elif self.p.exp_note == 'spfs' and self.p.few_shot_op is not None:
            self.model.ops = self.model.generate_single_path_agg_comb_act_few_shot_comp(self.p.few_shot_op)
            # print(1)

    def spos_train_supernet_ps2(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        log_root = f'{save_root}/log'
        os.makedirs(log_root, exist_ok=True)
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        self.subgraph_selector_optimizer = torch.optim.Adam(
            self.subgraph_selector.parameters(), lr=self.p.ss_lr, weight_decay=self.p.l2)
        self.logger = get_logger(f'{log_root}/', f'train_supernet')
        if self.p.weight_sharing:
            save_model_path = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{"_".join(save_root.split("/")[-1].split("_")[:-3])}/400.pt'
            save_ss_path = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{"_".join(save_root.split("/")[-1].split("_")[:-3])}/400_ss.pt'
            print(save_model_path)
            print(save_ss_path)
            self.model.load_state_dict(torch.load(save_model_path))
            self.subgraph_selector.load_state_dict(torch.load(save_ss_path))
        for epoch in range(1, self.p.max_epochs+1):
            start_time = time.time()
            train_loss = self.architecture_search_spos_ps2_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', 'spos_train_supernet_ps2')
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
            # self.scheduler.step(train_loss)
            wandb.log({
                "train_loss": train_loss,
                "valid_loss": valid_loss
            })
            if epoch % 100 == 0:
                torch.save(self.model.state_dict(), f'{save_root}/{epoch}.pt')
                torch.save(self.subgraph_selector.state_dict(), f'{save_root}/{epoch}_ss.pt')

    def architecture_search_spos_ps2_epoch(self):
        self.model.train()
        self.subgraph_selector.train()
        loss_list = []
        train_iter = self.data_iter['train_rel']
        valid_iter = self.data_iter['valid_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            self.generate_single_path()
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector,
                                           search_algorithm=self.p.ss_search_algorithm)

            # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
            # pred = self.model(g, subj, obj)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                train_loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                train_loss = self.model.calc_loss(pred, labels)
            train_loss.backward()
            if self.p.clip_grad:
                clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.p.alpha_mode == 'train_loss':
                self.subgraph_selector_optimizer.step()
            elif self.p.alpha_mode == 'valid_loss':
                self.optimizer.zero_grad()
                self.subgraph_selector_optimizer.zero_grad()
                batch = next(iter(valid_iter))
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector,
                                               search_algorithm=self.p.ss_search_algorithm)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    valid_loss = self.model.calc_loss(pred, labels, pos_neg)
                else:
                    valid_loss = self.model.calc_loss(pred, labels)
                valid_loss.backward()
                self.subgraph_selector_optimizer.step()
            loss_list.append(train_loss.item())
        loss = np.mean(loss_list)
        return loss

    def predict_spos_search_ps2(self, split, mode, eval_mode=None, spos_mode='train_supernet'):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        self.subgraph_selector.eval()
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(test_iter):
                if spos_mode == 'train_supernet':
                    self.generate_single_path()
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                pred = self.model.compute_pred(hidden_all_ent, subj, obj, self.subgraph_selector, mode='argmax',
                                               search_algorithm=self.p.ss_search_algorithm)
                # print(hidden_all_ent.size()) # [num_ent, encoder_layer, dim]
                # pred = self.model(g, subj, obj, mode=eval_mode)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            # dict_res = metrics.classification_report(pos_labels, pos_scores, output_dict=True, zero_division=1)
            # results['macro_f1_per_class'] = get_f1_score_list(dict_res)
            # results['acc_per_class'] = get_acc_list(dict_res)
            loss = np.mean(loss_list)
        return results, loss

    def spos_arch_search_ps2(self):
        res_list = []
        sorted_list = []
        exp_note = '_' + self.p.exp_note if self.p.exp_note is not None else ''
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        # save_root = f'{self.prj_path}/checkpoints/{self.p.dataset}/{self.p.search_mode}'
        # save_model_path = f'{save_root}/{self.p.name}.pt'
        # save_ss_path = f'{save_root}/{self.p.name}_ss.pt'
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        if self.p.exp_note == 'spfs':
            epoch_list = [400]
            weight_sharing = '_ws'
        else:
            epoch_list = [800, 700,600]
            weight_sharing = ''
        for save_epoch in epoch_list:
            # try:
            #     self.model.load_state_dict(torch.load(f'{save_root}/{save_epoch}.pt'))
            #     self.subgraph_selector.load_state_dict(torch.load(f'{save_root}/{save_epoch}_ss.pt'))
            # except:
            #     continue
            self.logger = get_logger(f'{save_root}/log/', f'{save_epoch}_arch_search{exp_note}')
            # res_root = f'{self.prj_path}/search_res/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
            # os.makedirs(res_root, exist_ok=True)
            valid_loss_searched_arch_res = dict()
            valid_f1_searched_arch_res = dict()
            valid_auroc_searched_arch_res = dict()
            t_start = time.time()
            for epoch in range(0, self.p.spos_arch_sample_num):
                for sample_idx in range(1, self.p.asng_sample_num+1):
                    self.generate_single_path()
                    arch = "||".join(self.model.ops)
                    if self.p.exp_note == 'spfs':
                        few_shot_op = self.model.ops[0]
                    else:
                        few_shot_op = ''
                    self.model.load_state_dict(torch.load(f'{save_root}{exp_note}_{few_shot_op}{weight_sharing}/{save_epoch}.pt'))
                    self.subgraph_selector.load_state_dict(torch.load(f'{save_root}{exp_note}_{few_shot_op}{weight_sharing}/{save_epoch}_ss.pt'))
                    val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search_ps2')
                    test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search_ps2')
                    valid_loss_searched_arch_res.setdefault(arch, valid_loss)
                    self.logger.info(f'[Epoch {epoch*self.p.asng_sample_num+sample_idx}]: Path:{arch}')
                    if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                        self.logger.info(
                            f"[Epoch {epoch*self.p.asng_sample_num+sample_idx}]: Valid Loss: {valid_loss:.5}, Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                        self.logger.info(
                            f"[Epoch {epoch*self.p.asng_sample_num+sample_idx}]: Test Loss: {test_loss:.5}, Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
                        valid_auroc_searched_arch_res.setdefault(arch, val_results['auroc'])
                        sorted_list.append(val_results['auroc'])
                    else:
                        self.logger.info(
                            f"[Epoch {epoch*self.p.asng_sample_num+sample_idx}]: Valid Loss: {valid_loss:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid ACC: {val_results['acc']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                        self.logger.info(
                            f"[Epoch {epoch*self.p.asng_sample_num+sample_idx}]: Test Loss: {test_loss:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test ACC: {test_results['acc']:.5}, Test Cohen: {test_results['kappa']:.5}")
                        valid_f1_searched_arch_res.setdefault(arch, val_results['macro_f1'])
                        sorted_list.append(val_results['macro_f1'])
                res_list.append(sorted(sorted_list, reverse=True)[:self.p.asng_sample_num])
            with open(f"{save_root}/topK_{save_epoch}{exp_note}.pkl", "wb") as f:
                pickle.dump(res_list, f)
            t_end = time.time()
            search_time = (t_end - t_start)

            search_time = search_time / 3600
            self.logger.info(f'The search process costs {search_time:.2f}h.')
            import csv
            with open(f'{save_root}/valid_loss_{save_epoch}{exp_note}.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['arch', 'valid loss'])
                valid_loss_searched_arch_res_sorted = sorted(valid_loss_searched_arch_res.items(), key=lambda x :x[1])
                res = valid_loss_searched_arch_res_sorted
                for i in range(len(res)):
                    writer.writerow([res[i][0], res[i][1]])
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                with open(f'{save_root}/valid_auroc_{save_epoch}{exp_note}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid auroc'])
                    valid_auroc_searched_arch_res_sorted = sorted(valid_auroc_searched_arch_res.items(), key=lambda x: x[1],
                                                               reverse=True)
                    res = valid_auroc_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])
            else:
                with open(f'{save_root}/valid_f1_{save_epoch}{exp_note}.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid f1'])
                    valid_f1_searched_arch_res_sorted = sorted(valid_f1_searched_arch_res.items(), key=lambda x: x[1], reverse=True)
                    res = valid_f1_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])

    def joint_spos_ps2_fine_tune(self):
        self.best_valid_metric = 0.0
        self.best_test_metric = {}
        arch_rank = 1
        exp_note = '_' + self.p.exp_note if self.p.exp_note is not None else ''
        for save_epoch in [400]:
            self.save_epoch = save_epoch
            save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
            res_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/res'
            log_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/log'
            tmp_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/tmp'
            os.makedirs(res_root, exist_ok=True)
            os.makedirs(log_root, exist_ok=True)
            os.makedirs(tmp_root, exist_ok=True)
            print(save_root)
            self.logger = get_logger(f'{log_root}/', f'fine_tune_e{save_epoch}_vmtop{arch_rank}{exp_note}')
            metric = 'auroc' if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset else 'f1'
            with open(f'{save_root}/valid_{metric}_{self.save_epoch}{exp_note}_ng.csv', 'r') as csv_file:
                reader = csv.reader(csv_file)
                for _ in range(arch_rank):
                    next(reader) # rank 1 is to skip head of csv
                self.p.genotype = next(reader)[0]
                self.model.ops = self.p.genotype.split("||")
                if self.p.exp_note == 'spfs':
                    weight_sharing = '_ws'
                    few_shot_op = self.model.ops[0]
                else:
                    weight_sharing = ''
                    few_shot_op = ''
                self.ss_path = f'{save_root}{exp_note}_{few_shot_op}{weight_sharing}/{self.save_epoch}_ss.pt'
            study = optuna.create_study(directions=["maximize"])
            study.optimize(self.spfs_fine_tune_each, n_trials=self.p.tune_sample_num)
            self.p.lr = 10 ** study.best_params["learning_rate"]
            self.p.l2 = 10 ** study.best_params["weight_decay"]
            with open(f'{res_root}/fine_tune_e{save_epoch}_vmtop{arch_rank}{exp_note}.txt', "w") as f1:
                f1.write(f'{self.p.__dict__}\n')
                f1.write(f'Valid performance: {study.best_value}\n')
                f1.write(f'Test performance: {self.best_test_metric}')

    def joint_spos_fine_tune(self, parameter):
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        self.p.lr = 10 ** parameter['learning_rate']
        self.p.l2 = 10 ** parameter['weight_decay']
        run = self.reinit_wandb()
        run.finish()
        return {'loss': -self.p.lr, 'test_metric':self.p.l2, "status": STATUS_OK}

    def spfs_fine_tune_each(self, trial):
        exp_note = '_' + self.p.exp_note if self.p.exp_note is not None else ''
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        learning_rate = trial.suggest_float("learning_rate", -3.05, -2.95)
        weight_decay = trial.suggest_float("weight_decay", -5, -3)
        self.p.lr = 10 ** learning_rate
        self.p.l2 = 10 ** weight_decay
        self.model = self.get_model().to("cuda:0")
        print(type(self.model).__name__)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        save_model_path = f'{save_root}/tmp/fine_tune_e{self.save_epoch}{exp_note}.pt'
        self.subgraph_selector.load_state_dict(torch.load(str(self.ss_path)))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.2, patience=10, verbose=True)
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch_fine_tune()
            val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search_ps2')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_model_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_model_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_auroc": self.best_val_auroc})
                self.scheduler.step(self.best_val_auroc)
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_f1": self.best_val_f1})
                self.scheduler.step(self.best_val_f1)
            if self.early_stop_cnt == 15:
                self.logger.info("Early stop!")
                break
        self.load_model(save_model_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search_ps2')
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            # wandb.log({
            #     "test_auroc": test_results['auroc'],
            #     "test_auprc": test_results['auprc'],
            #     "test_ap": test_results['ap']
            # })
            # run.finish()
            if self.best_val_auroc > self.best_valid_metric:
                self.best_valid_metric = self.best_val_auroc
                self.best_test_metric = test_results
            return self.best_val_auroc
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            # wandb.log({
            #     "test_acc": test_results['acc'],
            #     "test_f1": test_results['macro_f1'],
            #     "test_cohen": test_results['kappa']
            # })
            # run.finish()
            if self.best_val_f1 > self.best_valid_metric:
                self.best_valid_metric = self.best_val_f1
                self.best_test_metric = test_results
            return self.best_val_f1

    def spos_fine_tune(self):
        self.best_valid_metric = 0.0
        self.best_test_metric = {}
        arch_rank = 1
        for save_epoch in [800]:
            self.save_epoch = save_epoch
            save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
            print(save_root)
            self.logger = get_logger(f'{save_root}/', f'{save_epoch}_finu_tune_vmtop{arch_rank}')
            metric = 'auroc' if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset else 'f1'
            with open(f'{save_root}/valid_{metric}_{self.save_epoch}.csv', 'r') as csv_file:
                reader = csv.reader(csv_file)
                for _ in range(arch_rank):
                    next(reader) # rank 1 is to skip head of csv
                self.p.genotype = next(reader)[0]
                self.model.ops = self.p.genotype.split("||")
            study = optuna.create_study(directions=["maximize"])
            study.optimize(self.spos_fine_tune_each, n_trials=self.p.tune_sample_num)
            self.p.lr = 10 ** study.best_params["learning_rate"]
            self.p.l2 = 10 ** study.best_params["weight_decay"]
            save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
            with open(f'{save_root}/{save_epoch}_tune_res_vmtop{arch_rank}.txt', "w") as f1:
                f1.write(f'{self.p.__dict__}\n')
                f1.write(f'Valid performance: {study.best_value}\n')
                f1.write(f'Test performance: {self.best_test_metric}')

    def spos_fine_tune_each(self, trial):
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        learning_rate = trial.suggest_float("learning_rate", -3.05, -2.95)
        weight_decay = trial.suggest_float("weight_decay", -5, -3)
        self.p.lr = 10 ** learning_rate
        self.p.l2 = 10 ** weight_decay
        # run = self.reinit_wandb()
        self.model = self.get_model().to("cuda:0")
        print(type(self.model).__name__)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
        # for save_epoch in [100, 200, 300, 400]:
        save_model_path = f'{save_root}/{self.save_epoch}_tune.pt'
        # self.model.load_state_dict(torch.load(str(supernet_path)))
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_model_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_model_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_auroc": self.best_val_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                # wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                #            "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        self.load_model(save_model_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search')
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            # wandb.log({
            #     "test_auroc": test_results['auroc'],
            #     "test_auprc": test_results['auprc'],
            #     "test_ap": test_results['ap']
            # })
            # run.finish()
            if self.best_val_auroc > self.best_valid_metric:
                self.best_valid_metric = self.best_val_auroc
                self.best_test_metric = test_results
            return self.best_val_auroc
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            # wandb.log({
            #     "test_acc": test_results['acc'],
            #     "test_f1": test_results['macro_f1'],
            #     "test_cohen": test_results['kappa']
            # })
            # run.finish()
            if self.best_val_f1 > self.best_valid_metric:
                self.best_valid_metric = self.best_val_f1
                self.best_test_metric = test_results
            return self.best_val_f1

    def train_mix_hop(self):
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.train_mode}/{self.p.name}/'
        os.makedirs(save_root, exist_ok=True)
        self.logger = get_logger(f'{save_root}/', f'train')
        save_path = f'{save_root}/model_weight.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.train_epoch_mix_hop()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='normal_mix_hop')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    self.save_model(save_path)
                    self.early_stop_cnt = 0
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_auroc": self.best_val_auroc})
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss,
                           "best_valid_f1": self.best_val_f1})
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        # self.logger.info(vars(self.p))
        self.load_model(save_path)
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='normal_mix_hop')
        end = time.time()
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            wandb.log({
                "test_auroc": test_results['auroc'],
                "test_auprc": test_results['auprc'],
                "test_ap": test_results['ap']
            })
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            wandb.log({
                "test_acc": test_results['acc'],
                "test_f1": test_results['macro_f1'],
                "test_cohen": test_results['kappa']
            })

    def train_epoch_mix_hop(self):
        self.model.train()
        losses = []
        train_iter = self.data_iter['train_rel']
        # train_bar = tqdm(train_iter, ncols=0)
        for step, batch in enumerate(train_iter):
            if self.p.input_type == 'subgraph':
                g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
            else:
                triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                g = self.g.to("cuda:0")
            subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
            hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
            mix_hop_index = self.transform_hop_index()
            pred = self.model.compute_mix_hop_pred(hidden_all_ent, subj, obj, mix_hop_index)
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                pos_neg = batch[2].to("cuda:0")
                loss = self.model.calc_loss(pred, labels, pos_neg)
            else:
                loss = self.model.calc_loss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            if self.p.clip_grad:
                clip_grad_norm_(self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()
            losses.append(loss.item())
        loss = np.mean(losses)
        return loss

    def transform_hop_index(self):
        ij = self.p.exp_note.split("_")
        return self.p.n_layer * (int(ij[0]) - 1) + int(ij[1]) - 1

    def predict_mix_hop(self, split, mode):
        loss_list = []
        pos_scores = []
        pos_labels = []
        pred_class = {}
        self.model.eval()
        with torch.no_grad():
            results = dict()
            eval_iter = self.data_iter[f'{split}_rel{mode}']
            for step, batch in enumerate(eval_iter):
                if self.p.input_type == 'subgraph':
                    g, triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0"), batch[2].to("cuda:0")
                else:
                    triplets, labels = batch[0].to("cuda:0"), batch[1].to("cuda:0")
                    g = self.g.to("cuda:0")
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                hidden_all_ent, all_ent = self.model.forward_search(g, mode=self.p.input_type)  # [batch_size, num_ent]
                mix_hop_index = self.transform_hop_index()
                pred = self.model.compute_mix_hop_pred(hidden_all_ent, subj, obj, mix_hop_index)
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    pos_neg = batch[2].to("cuda:0")
                    eval_loss = self.model.calc_loss(pred, labels, pos_neg)
                    m = torch.nn.Sigmoid()
                    pred = m(pred)
                    labels = labels.detach().to('cpu').numpy()
                    preds = pred.detach().to('cpu').numpy()
                    pos_neg = pos_neg.detach().to('cpu').numpy()
                    for (label_ids, pred, label_t) in zip(labels, preds, pos_neg):
                        for i, (l, p) in enumerate(zip(label_ids, pred)):
                            if l == 1:
                                if i in pred_class:
                                    pred_class[i]['pred'] += [p]
                                    pred_class[i]['l'] += [label_t]
                                    pred_class[i]['pred_label'] += [1 if p > 0.5 else 0]
                                else:
                                    pred_class[i] = {'pred': [p], 'l': [label_t], 'pred_label': [1 if p > 0.5 else 0]}
                else:
                    eval_loss = self.model.calc_loss(pred, labels)
                    pos_labels += rel.to('cpu').numpy().flatten().tolist()
                    pos_scores += torch.argmax(pred, dim=1).cpu().flatten().tolist()
                loss_list.append(eval_loss.item())
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                roc_auc = [metrics.roc_auc_score(pred_class[l]['l'], pred_class[l]['pred']) for l in pred_class]
                prc_auc = [metrics.average_precision_score(pred_class[l]['l'], pred_class[l]['pred']) for l in
                           pred_class]
                ap = [metrics.accuracy_score(pred_class[l]['l'], pred_class[l]['pred_label']) for l in pred_class]
                results['auroc'] = np.mean(roc_auc)
                results['auprc'] = np.mean(prc_auc)
                results['ap'] = np.mean(ap)
            else:
                results['acc'] = metrics.accuracy_score(pos_labels, pos_scores)
                results['macro_f1'] = metrics.f1_score(pos_labels, pos_scores, average='macro')
                results['kappa'] = metrics.cohen_kappa_score(pos_labels, pos_scores)
            loss = np.mean(loss_list)
        return results, loss

    def joint_random_ps2_each(self, trial):
        genotype_space = []
        for i in range(self.p.n_layer):
            genotype_space.append(trial.suggest_categorical("mess"+ str(i), COMP_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("agg"+ str(i), AGG_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("comb"+ str(i), COMB_PRIMITIVES))
            genotype_space.append(trial.suggest_categorical("act"+ str(i), ACT_PRIMITIVES))
        self.best_val_f1 = 0.0
        self.best_val_auroc = 0.0
        self.early_stop_cnt = 0
        # self.best_valid_metric, self.best_test_metric = 0.0, {}
        self.p.genotype = "||".join(genotype_space)
        self.model = self.get_model().to("cuda:0")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        self.subgraph_selector_optimizer = torch.optim.Adam(
            self.subgraph_selector.parameters(), lr=self.p.ss_lr, weight_decay=self.p.l2)
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        os.makedirs(save_root, exist_ok=True)
        save_model_path = f'{save_root}/model.pt'
        save_ss_path = f'{save_root}/model_ss.pt'
        save_model_best_path = f'{save_root}/model_best.pt'
        save_ss_best_path = f'{save_root}/model_best_ss.pt'
        for epoch in range(self.p.max_epochs):
            start_time = time.time()
            train_loss = self.search_epoch()
            val_results, valid_loss = self.evaluate_epoch('valid', mode='ps2')
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                if val_results['auroc'] > self.best_val_auroc:
                    self.best_val_results = val_results
                    self.best_val_auroc = val_results['auroc']
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), str(save_model_path))
                    torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
                    self.early_stop_cnt = 0
                    # self.logger.info("Update best valid auroc!")
                else:
                    self.early_stop_cnt += 1
            else:
                if val_results['macro_f1'] > self.best_val_f1:
                    self.best_val_results = val_results
                    self.best_val_f1 = val_results['macro_f1']
                    self.best_epoch = epoch
                    torch.save(self.model.state_dict(), str(save_model_path))
                    torch.save(self.subgraph_selector.state_dict(), str(save_ss_path))
                    self.early_stop_cnt = 0
                    # self.logger.info("Update best valid f1!")
                else:
                    self.early_stop_cnt += 1
            self.logger.info(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid Loss: {valid_loss:.5}, Cost: {time.time() - start_time:.2f}s")
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best AUROC: {self.best_val_auroc:.5}")
            else:
                self.logger.info(
                    f"[Epoch {epoch}]: Valid ACC: {val_results['acc']:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                self.logger.info(
                    f"[Epoch {epoch}]: Best Macro F1: {self.best_val_f1:.5}")
            if self.early_stop_cnt == 10:
                self.logger.info("Early stop!")
                break
        self.model.load_state_dict(torch.load(str(save_model_path)))
        self.subgraph_selector.load_state_dict(torch.load(str(save_ss_path)))
        self.logger.info(
            f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        self.logger.info(f'{self.p.genotype}')
        start = time.time()
        test_results, test_loss = self.evaluate_epoch('test', mode='ps2')
        end = time.time()
        if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
            self.logger.info(
                f"[Inference]: Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
            if self.best_val_auroc > self.best_valid_metric:
                self.best_valid_metric = self.best_val_auroc
                self.best_test_metric = test_results
                torch.save(self.model.state_dict(), str(save_model_best_path))
                torch.save(self.subgraph_selector.state_dict(), str(save_ss_best_path))
            with open(f'{save_root}/random_ps2_arch_list.csv', "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.p.genotype, self.best_val_auroc, test_results['auroc']])
            return self.best_val_auroc
        else:
            self.logger.info(
                f"[Inference]: Test ACC: {test_results['acc']:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test Cohen: {test_results['kappa']:.5}")
            if self.best_val_f1 > self.best_valid_metric:
                self.best_valid_metric = self.best_val_f1
                self.best_test_metric = test_results
                torch.save(self.model.state_dict(), str(save_model_best_path))
                torch.save(self.subgraph_selector.state_dict(), str(save_ss_best_path))
            with open(f'{save_root}/random_ps2_arch_list.csv', "a") as f:
                writer = csv.writer(f)
                writer.writerow([self.p.genotype, self.best_val_f1, test_results['macro_f1']])
            return self.best_val_f1

    def joint_random_ps2(self):
        self.best_valid_metric = 0.0
        self.best_test_metric = {}
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}/'
        os.makedirs(save_root, exist_ok=True)
        print(save_root)
        self.logger = get_logger(f'{save_root}/', f'random_ps2')
        study = optuna.create_study(directions=["maximize"], sampler=RandomSampler())
        study.optimize(self.joint_random_ps2_each, n_trials=self.p.baseline_sample_num)
        with open(f'{save_root}/random_ps2_res.txt', "w") as f1:
            f1.write(f'{self.p.__dict__}\n')
            f1.write(f'{self.p.genotype}\n')
            f1.write(f'Valid performance: {study.best_value}\n')
            f1.write(f'Test performance: {self.best_test_metric}')

    def spos_arch_search_ps2_ng(self):
        res_list = []
        sorted_list = []
        arch_list = []
        for _ in range(self.p.n_layer):
            arch_list.append(len(COMP_PRIMITIVES))
            arch_list.append(len(AGG_PRIMITIVES))
            arch_list.append(len(COMB_PRIMITIVES))
            arch_list.append(len(ACT_PRIMITIVES))
        asng = CategoricalASNG(np.array(arch_list), alpha=1.5, delta_init=1)
        exp_note = '_' + self.p.exp_note if self.p.exp_note is not None else ''
        self.subgraph_selector = SubgraphSelector(self.p).to("cuda:0")
        save_root = f'{self.prj_path}/exp/{self.p.dataset}/{self.p.search_mode}/{self.p.name}'
        if self.p.exp_note == 'spfs':
            epoch_list = [400]
            weight_sharing = '_ws'
        else:
            epoch_list = [800, 700,600]
            weight_sharing = ''
        for save_epoch in epoch_list:
            valid_metric = 0.0
            self.logger = get_logger(f'{save_root}/', f'{save_epoch}_arch_search_ng{exp_note}')
            valid_loss_searched_arch_res = dict()
            valid_f1_searched_arch_res = dict()
            valid_auroc_searched_arch_res = dict()
            search_time = 0.0
            t_start = time.time()
            for epoch in range(1, self.p.spos_arch_sample_num + 1):
                Ms = []
                ma_structs = []
                scores = []
                for i in range(self.p.asng_sample_num):
                    M = asng.sampling()
                    struct = np.argmax(M, axis=1)
                    Ms.append(M)
                    ma_structs.append(list(struct))
                    # print(list(struct))
                    self.generate_single_path_ng(list(struct))
                    arch = "||".join(self.model.ops)
                    if self.p.exp_note == 'spfs':
                        few_shot_op = self.model.ops[0]
                    else:
                        few_shot_op = ''
                    self.model.load_state_dict(
                        torch.load(f'{save_root}{exp_note}_{few_shot_op}{weight_sharing}/{save_epoch}.pt'))
                    self.subgraph_selector.load_state_dict(
                        torch.load(f'{save_root}{exp_note}_{few_shot_op}{weight_sharing}/{save_epoch}_ss.pt'))
                    val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search_ps2')
                    test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search_ps2')
                    valid_loss_searched_arch_res.setdefault(arch, valid_loss)
                    self.logger.info(f'[Epoch {epoch}, {i}-th arch]: Path:{arch}')
                    if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                        self.logger.info(
                            f"[Epoch {epoch}, {i}-th arch]: Valid Loss: {valid_loss:.5}, Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                        self.logger.info(
                            f"[Epoch {epoch}, {i}-th arch]: Test Loss: {test_loss:.5}, Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
                        scores.append(val_results['auroc'])
                        sorted_list.append(val_results['auroc'])
                        valid_auroc_searched_arch_res.setdefault(arch, val_results['auroc'])
                    else:
                        self.logger.info(
                            f"[Epoch {epoch}, {i}-th arch]: Valid Loss: {valid_loss:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid ACC: {val_results['acc']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                        self.logger.info(
                            f"[Epoch {epoch}, {i}-th arch]: Test Loss: {test_loss:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test ACC: {test_results['acc']:.5}, Test Cohen: {test_results['kappa']:.5}")
                        valid_f1_searched_arch_res.setdefault(arch, val_results['macro_f1'])
                        scores.append(val_results['macro_f1'])
                        sorted_list.append(val_results['macro_f1'])
                res_list.append(sorted(sorted_list, reverse=True)[:self.p.asng_sample_num])
                asng.update(np.array(Ms), -np.array(scores), True)
                best_struct = list(asng.theta.argmax(axis=1))
                self.generate_single_path_ng(best_struct)
                arch = "||".join(self.model.ops)
                val_results, valid_loss = self.evaluate_epoch('valid', 'spos_arch_search_ps2')
                test_results, test_loss = self.evaluate_epoch('test', 'spos_arch_search_ps2')
                self.logger.info(f'Path:{arch}')
                if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                    self.logger.info(
                        f"Valid Loss: {valid_loss:.5}, Valid AUROC: {val_results['auroc']:.5}, Valid AUPRC: {val_results['auprc']:.5}, Valid AP@50: {val_results['ap']:.5}")
                    self.logger.info(
                        f"Test Loss: {test_loss:.5}, Test AUROC: {test_results['auroc']:.5}, Test AUPRC: {test_results['auprc']:.5}, Test AP@50: {test_results['ap']:.5}")
                else:
                    self.logger.info(
                        f"Valid Loss: {valid_loss:.5}, Valid Macro F1: {val_results['macro_f1']:.5}, Valid ACC: {val_results['acc']:.5}, Valid Cohen: {val_results['kappa']:.5}")
                    self.logger.info(
                        f"Test Loss: {test_loss:.5}, Test Macro F1: {test_results['macro_f1']:.5}, Test ACC: {test_results['acc']:.5}, Test Cohen: {test_results['kappa']:.5}")
            with open(f"{save_root}/topK_{save_epoch}{exp_note}_ng.pkl", "wb") as f:
                pickle.dump(res_list, f)
            t_end = time.time()
            search_time = (t_end - t_start)

            search_time = search_time / 3600
            self.logger.info(f'The search process costs {search_time:.2f}h.')
            import csv
            with open(f'{save_root}/valid_loss_{save_epoch}{exp_note}_ng.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['arch', 'valid loss'])
                valid_loss_searched_arch_res_sorted = sorted(valid_loss_searched_arch_res.items(), key=lambda x :x[1])
                res = valid_loss_searched_arch_res_sorted
                for i in range(len(res)):
                    writer.writerow([res[i][0], res[i][1]])
            if 'twosides' in self.p.dataset or 'ogbl_biokg' in self.p.dataset:
                with open(f'{save_root}/valid_auroc_{save_epoch}{exp_note}_ng.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid auroc'])
                    valid_auroc_searched_arch_res_sorted = sorted(valid_auroc_searched_arch_res.items(), key=lambda x: x[1],
                                                               reverse=True)
                    res = valid_auroc_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])
            else:
                with open(f'{save_root}/valid_f1_{save_epoch}{exp_note}_ng.csv', 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['arch', 'valid f1'])
                    valid_f1_searched_arch_res_sorted = sorted(valid_f1_searched_arch_res.items(), key=lambda x: x[1], reverse=True)
                    res = valid_f1_searched_arch_res_sorted
                    for i in range(len(res)):
                        writer.writerow([res[i][0], res[i][1]])
    def generate_single_path_ng(self, struct):
        single_path = []
        for ops_index, index in enumerate(struct):
            if ops_index % 4 == 0:
                single_path.append(COMP_PRIMITIVES[index])
            elif ops_index % 4 == 1:
                single_path.append(AGG_PRIMITIVES[index])
            elif ops_index % 4 == 2:
                single_path.append(COMB_PRIMITIVES[index])
            elif ops_index % 4 == 3:
                single_path.append(ACT_PRIMITIVES[index])
        self.model.ops = single_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run',
                        help='Set run name for saving/restoring models')
    parser.add_argument('--dataset', default='drugbank',
                        help='Dataset to use, default: FB15k-237')
    parser.add_argument('--input_type', type=str, default='allgraph', choices=['subgraph', 'allgraph'])
    parser.add_argument('--score_func', dest='score_func', default='none',
                        help='Score Function for Link prediction')
    parser.add_argument('--opn', dest='opn', default='corr',
                        help='Composition Operation to be used in CompGCN')

    parser.add_argument('--batch', dest='batch_size',
                        default=256, type=int, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs',
                        type=int, default=500, help='Number of epochs')
    parser.add_argument('--l2', type=float, default=5e-4,
                        help='L2 Regularization for Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Starting Learning Rate')
    parser.add_argument('--lbl_smooth', dest='lbl_smooth',
                        type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345,
                        type=int, help='Seed for randomization')

    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')
    parser.add_argument('--bias', dest='bias', action='store_true',
                        help='Whether to use bias in the model')

    parser.add_argument('--num_bases', dest='num_bases', default=-1, type=int,
                        help='Number of basis relation vectors to use')
    parser.add_argument('--init_dim', dest='init_dim', default=100, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200,
                        type=int, help='Number of hidden units in GCN')
    parser.add_argument('--embed_dim', dest='embed_dim', default=None, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('--n_layer', dest='n_layer', default=1,
                        type=int, help='Number of GCN Layers to use')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.1,
                        type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('--hid_drop', dest='hid_drop',
                        default=0.3, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('--conve_hid_drop', dest='conve_hid_drop', default=0.3, type=float,
                        help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop',
                        default=0.2, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--input_drop', dest='input_drop', default=0.2,
                        type=float, help='ConvE: Stacked Input Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20,
                        type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10,
                        type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7,
                        type=int, help='ConvE: Kernel size to use')

    parser.add_argument('--gamma', dest='gamma', default=9.0,
                        type=float, help='TransE: Gamma to use')

    parser.add_argument('--rat', action='store_true',
                        default=False, help='random adacency tensors')
    parser.add_argument('--wni', action='store_true',
                        default=False, help='without neighbor information')
    parser.add_argument('--wsi', action='store_true',
                        default=False, help='without self-loop information')
    parser.add_argument('--ss', dest='ss', default=-1,
                        type=int, help='sample size (sample neighbors)')
    parser.add_argument('--nobn', action='store_true',
                        default=False, help='no use of batch normalization in aggregation')
    parser.add_argument('--noltr', action='store_true',
                        default=False, help='no use of linear transformations for relation embeddings')

    parser.add_argument('--encoder', dest='encoder',
                        default='compgcn', type=str, help='which encoder to use')

    # for lte models
    parser.add_argument('--x_ops', dest='x_ops', default="")
    parser.add_argument('--r_ops', dest='r_ops', default="")

    parser.add_argument("--ss_num_layer", default=2, type=int)
    parser.add_argument("--ss_input_dim", default=200, type=int)
    parser.add_argument("--ss_hidden_dim", default=200, type=int)
    parser.add_argument("--ss_lr", default=0.001, type=float)
    parser.add_argument('--train_mode', default='', type=str,
                        choices=["train", "tune", "DEBUG", "vis_hop", "inference", "vis_class","joint_tune","spos_tune","vis_hop_pred","vis_rank_ccorelation","vis_rank_ccorelation_spfs"])
    parser.add_argument("--ss_model_path", type=str)
    parser.add_argument("--ss_search_algorithm", default='darts', type=str)
    parser.add_argument("--search_algorithm", default='darts', type=str)
    parser.add_argument("--temperature", default=0.07, type=float)
    parser.add_argument("--temperature_min", default=0.005, type=float)
    parser.add_argument("--lr_min", type=float, default=0.001)
    parser.add_argument("--arch_lr", default=0.001, type=float)
    parser.add_argument("--arch_lr_min", default=0.001, type=float)
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
    parser.add_argument("--w_update_epoch", type=int, default=1)
    parser.add_argument("--alpha_mode", type=str, default='valid_loss')
    parser.add_argument('--loc_mean', type=float, default=10.0, help='initial mean value to generate the location')
    parser.add_argument('--loc_std', type=float, default=0.01, help='initial std to generate the location')
    parser.add_argument("--genotype", type=str, default=None)
    parser.add_argument("--baseline_sample_num", type=int, default=200)
    parser.add_argument("--cos_temp", action='store_true', default=False, help='temp decay')
    parser.add_argument("--spos_arch_sample_num", default=1000, type=int)
    parser.add_argument("--tune_sample_num", default=10, type=int)

    # subgraph config
    parser.add_argument("--subgraph_type", type=str, default='seal')
    parser.add_argument("--subgraph_hop", type=int, default=2)
    parser.add_argument("--subgraph_edge_sample_ratio", type=float, default=1)
    parser.add_argument("--subgraph_is_saved", type=bool, default=True)
    parser.add_argument("--subgraph_max_num_nodes", type=int, default=100)
    parser.add_argument("--subgraph_sample_type", type=str, default='enclosing_subgraph')
    parser.add_argument("--save_mode", type=str, default='graph')
    parser.add_argument("--num_neg_samples_per_link", type=int, default=0)

    # transformer config
    parser.add_argument("--d_model", type=int, default=100)
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=100)
    parser.add_argument("--transformer_dropout", type=float, default=0.1)
    parser.add_argument("--transformer_activation", type=str, default='relu')
    parser.add_argument("--concat_type", type=str, default='so')
    parser.add_argument("--graph_pooling_type", type=str, default='mean')

    parser.add_argument("--loss_type", type=str, default='ce')
    parser.add_argument("--eval_mode", type=str, default='rel')
    parser.add_argument("--wandb_project", type=str, default='')
    parser.add_argument("--search_mode", type=str, default='', choices=["ps2", "ps2_random", "", "arch_search", "arch_random","joint_search","arch_spos"])
    parser.add_argument("--add_reverse", action='store_true', default=False)
    parser.add_argument("--clip_grad", action='store_true', default=False)
    parser.add_argument("--fine_tune_with_implicit_subgraph", action='store_true', default=False)
    parser.add_argument("--combine_type", type=str, default='concat')
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument("--few_shot_op", type=str, default=None)
    parser.add_argument("--weight_sharing", action='store_true', default=False)

    parser.add_argument("--asng_sample_num", type=int, default=16)
    parser.add_argument("--arch_search_mode", type=str, default='random')

    args = parser.parse_args()
    opn = '_' + args.opn if args.encoder == 'compgcn' else ''
    reverse = '_add_reverse' if args.add_reverse else ''
    genotype = '_'+args.genotype if args.genotype is not None else ''
    input_type = '_'+args.input_type if args.input_type == 'subgraph' else ''
    exp_note = '_'+args.exp_note if args.exp_note is not None else ''
    num_bases = '_b'+str(args.num_bases) if args.num_bases!=-1 else ''
    alpha_mode = '_'+str(args.alpha_mode)
    few_shot_op = '_'+args.few_shot_op if args.few_shot_op is not None else ''
    weight_sharing = '_ws' if args.weight_sharing else ''
    ss_search_algorithm = '_snas' if args.ss_search_algorithm == 'snas' else ''

    if args.input_type == 'subgraph':
        args.name = 'seal'
    else:
        if args.train_mode in ['train', 'vis_hop_pred']:
            args.name = f'{args.encoder}{opn}_{args.score_func}_{args.combine_type}_train_layer{args.n_layer}_seed{args.seed}{num_bases}{reverse}{genotype}{exp_note}'
        elif args.search_mode in ['ps2']:
            args.name = f'{args.encoder}{opn}_{args.score_func}_{args.combine_type}_{args.search_mode}{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}{num_bases}{reverse}{genotype}{exp_note}'
        elif args.search_mode in ['arch_random']:
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_{args.search_mode}_layer{args.n_layer}_seed{args.seed}{reverse}{exp_note}'
        elif args.search_algorithm == 'spos_arch_search':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_spos_train_supernet_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}{exp_note}{few_shot_op}{weight_sharing}'
        elif args.search_mode == 'joint_search' and args.train_mode == 'vis_hop':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_spos_train_supernet_ps2{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}{reverse}{exp_note}{few_shot_op}{weight_sharing}'
        elif args.search_algorithm == 'spos_arch_search_ps2':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_spos_train_supernet_ps2{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}'
        elif args.search_mode == 'joint_search' and args.train_mode == 'spos_tune':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_spos_train_supernet_ps2{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}'
        elif args.search_mode == 'joint_search' and args.search_algorithm == 'random_ps2':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_{args.search_algorithm}{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}'
        elif args.search_algorithm == 'spos_train_supernet_ps2':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_{args.search_algorithm}{ss_search_algorithm}_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}{exp_note}{few_shot_op}{weight_sharing}'
        elif (args.search_algorithm == 'spos_train_supernet' and args.train_mode!='spos_tune') or args.train_mode == 'vis_rank_correlation' or args.train_mode == 'vis_rank_ccorelation_spfs':
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_{args.search_algorithm}_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}{exp_note}{few_shot_op}{weight_sharing}'
        else:
            args.name = f'{args.encoder}_{args.score_func}_{args.combine_type}_spos_train_supernet_layer{args.n_layer}_seed{args.seed}{reverse}{genotype}{exp_note}{few_shot_op}{weight_sharing}'

    args.embed_dim = args.k_w * \
                     args.k_h if args.embed_dim is None else args.embed_dim

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("HYPEROPT_FMIN_SEED", str(args.seed))

    runner = Runner(args)

    if args.search_mode == 'joint_search' and args.search_algorithm!='spos_arch_search_ps2' and args.train_mode!='spos_tune':
        wandb.init(
            project=args.wandb_project,
            config={
                "dataset": args.dataset,
                "encoder": args.encoder,
                "score_function": args.score_func,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "encoder_layer_num": args.n_layer,
                "epochs": args.max_epochs,
                "seed": args.seed,
                "init_dim": args.init_dim,
                "embed_dim": args.embed_dim,
                "loss_type": args.loss_type,
                "search_mode": args.search_mode,
                "combine_type": args.combine_type,
                "note": args.exp_note,
                "search_algorithm": args.search_algorithm,
                "weight_sharing": args.weight_sharing,
                "few_shot_op": args.few_shot_op,
                "ss_search_algorithm": args.ss_search_algorithm
            })

    if args.train_mode == 'train':
        if args.exp_note is not None:
            runner.train_mix_hop()
        else:
            runner.train()
    elif args.train_mode == 'tune' and args.search_mode in ['ps2', 'joint_search']:
        runner.fine_tune()
    elif args.train_mode == 'vis_hop' and args.search_mode in ['ps2', "ps2_random", "joint_search"]:
        runner.vis_hop_distrubution()
    elif args.train_mode == 'vis_hop_pred':
        runner.vis_hop_pred()
    elif args.train_mode == 'inference' and args.search_mode in ['ps2']:
        runner.inference()
    elif args.train_mode == 'vis_class':
        runner.vis_class_distribution()
    elif args.train_mode == 'joint_tune':
        runner.joint_tune()
    else:
        if args.search_mode == 'ps2_random':
            runner.random_search()
        elif args.search_mode == 'ps2':
            runner.ps2()
        elif args.search_mode == 'arch_search':
            if args.train_mode == 'spos_tune':
                runner.spos_fine_tune()
            elif args.train_mode in ["vis_rank_ccorelation"]:
                runner.vis_rank_ccorelation()
            elif args.train_mode in ["vis_rank_ccorelation_spfs"]:
                runner.vis_rank_ccorelation_spfs()
            elif args.search_algorithm in ["darts", "snas"]:
                runner.architecture_search()
            elif args.search_algorithm in ["spos_train_supernet"]:
                runner.spos_train_supernet()
            elif args.search_algorithm in ["spos_arch_search"]:
                runner.spos_arch_search()
        elif args.search_mode == 'arch_random':
            runner.arch_random_search()
        elif args.search_mode == 'joint_search':
            if args.search_algorithm in ["spos_train_supernet_ps2"]:
                runner.spos_train_supernet_ps2()
            elif args.search_algorithm in ["spos_arch_search_ps2"] and args.arch_search_mode == 'random':
                runner.spos_arch_search_ps2()
            elif args.search_algorithm in ["spos_arch_search_ps2"] and args.arch_search_mode == 'ng':
                runner.spos_arch_search_ps2_ng()
            elif args.train_mode in ["spos_tune"]:
                runner.joint_spos_ps2_fine_tune()
            elif args.search_algorithm == 'random_ps2':
                runner.joint_random_ps2()
            else:
                runner.joint_search()
