import torch
from torch import nn
import dgl
from model.rgcn_layer import RelGraphConv
from model.compgcn_layer import CompGCNCov
import torch.nn.functional as F
from dgl import NID, EID, readout_nodes
from dgl.nn.pytorch import GraphConv
import time


class GCNs(nn.Module):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., opn='mult', wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True, input_type='subgraph', loss_type='ce', add_reverse=True):
        super(GCNs, self).__init__()
        self.act = torch.tanh
        if loss_type == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            self.loss = nn.BCELoss(reduce=False)
        elif loss_type == 'bce_logits':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim
        self.conv_bias = conv_bias
        self.gcn_drop = gcn_drop
        self.opn = opn
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer
        self.input_type = input_type

        self.wni = wni

        self.encoder = encoder
        if input_type == 'subgraph':
            self.init_embed = self.get_param([self.num_ent, self.init_dim])
            # self.init_embed = nn.Embedding(self.num_ent+2, self.init_dim)
        else:
            self.init_embed = self.get_param([self.num_ent + 1, self.init_dim])
        if add_reverse:
            self.init_rel = self.get_param([self.num_rel * 2, self.init_dim])
            self.bias_rel = nn.Parameter(torch.zeros(self.num_rel * 2))
        else:
            self.init_rel = self.get_param([self.num_rel, self.init_dim])
            self.bias_rel = nn.Parameter(torch.zeros(self.num_rel))

        if encoder == 'compgcn':
            if n_layer < 3:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        add_reverse=add_reverse)
                self.conv2 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        add_reverse=add_reverse) if n_layer == 2 else None
            else:
                self.conv1 = CompGCNCov(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        add_reverse=add_reverse)
                self.conv2 = CompGCNCov(self.gcn_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, opn, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        add_reverse=add_reverse)
                self.conv3 = CompGCNCov(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop,
                                        opn, num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        add_reverse=add_reverse)
        elif encoder == 'rgcn':
            if n_layer < 3:
                self.conv1 = RelGraphConv(self.init_dim, self.gcn_dim, self.num_rel * 2, "bdd",
                                          num_bases=self.num_base, activation=self.act, self_loop=(not wsi),
                                          dropout=gcn_drop, wni=wni)
                self.conv2 = RelGraphConv(self.gcn_dim, self.embed_dim, self.num_rel * 2, "bdd", num_bases=self.num_base,
                                          activation=self.act, self_loop=(not wsi), dropout=gcn_drop,
                                          wni=wni) if n_layer == 2 else None
            else:
                self.conv1 = RelGraphConv(self.init_dim, self.gcn_dim, self.num_rel * 2, "bdd",
                                          num_bases=self.num_base, activation=self.act, self_loop=(not wsi),
                                          dropout=gcn_drop, wni=wni)
                self.conv2 = RelGraphConv(self.gcn_dim, self.gcn_dim, self.num_rel * 2, "bdd",
                                          num_bases=self.num_base, activation=self.act, self_loop=(not wsi),
                                          dropout=gcn_drop, wni=wni)
                self.conv3 = RelGraphConv(self.gcn_dim, self.embed_dim, self.num_rel * 2, "bdd", num_bases=self.num_base,
                                          activation=self.act, self_loop=(not wsi), dropout=gcn_drop,
                                          wni=wni) if n_layer == 2 else None
        elif encoder == 'gcn':
                self.conv1 = GraphConv(self.init_dim, self.gcn_dim, allow_zero_in_degree=True)
                self.conv2 = GraphConv(self.gcn_dim, self.gcn_dim, allow_zero_in_degree=True)

        self.bias = nn.Parameter(torch.zeros(self.num_ent))
        # self.bias_rel = nn.Parameter(torch.zeros(self.num_rel*2))

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def calc_loss(self, pred, label, pos_neg=None):
        if pos_neg is not None:
            m = nn.Sigmoid()
            score_pos = m(pred)
            targets_pos = pos_neg.unsqueeze(1)
            loss = self.loss(score_pos, label * targets_pos)
            return torch.sum(loss * label)
        return self.loss(pred, label)

    def forward_base(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations

        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                if self.n_layer < 3:
                    x = self.conv1(g, x, self.edge_type, self.edge_norm.unsqueeze(-1))
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x = self.conv2(
                        g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x = self.conv1(g, x, self.edge_type, self.edge_norm.unsqueeze(-1))
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x = self.conv2(g, x, self.edge_type, self.edge_norm.unsqueeze(-1))
                    x = drop1(x)
                    x = self.conv3(g, x, self.edge_type, self.edge_norm.unsqueeze(-1))
                    x = drop2(x)

                    # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of relations in this batch
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x

    def forward_base_search(self, g, subj, rel, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x_hidden = []
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x_hidden.append(x)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of relations in this batch
        rel_emb = torch.index_select(r, 0, rel)

        x_hidden = torch.stack(x_hidden, dim=1)

        return x_hidden

    def forward_base_rel(self, g, subj, obj, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations

        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of objects in this batch
        obj_emb = torch.index_select(x, 0, obj)

        return sub_emb, obj_emb, x, r

    def forward_base_rel_vis_hop(self, g, subj, obj, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x_hidden = []

        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x_hidden.append(x)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x_hidden.append(x)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        # filter out embeddings of subjects in this batch
        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of objects in this batch
        obj_emb = torch.index_select(x, 0, obj)
        x_hidden = torch.stack(x_hidden, dim=1)

        return sub_emb, obj_emb, x, r, x_hidden

    def forward_base_rel_search(self, g, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x_hidden = []
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x_hidden.append(x)
                    x, r = self.conv2(
                        g, x, r, self.edge_type, self.edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                    x_hidden.append(x)
                else:
                    x, r = self.conv1(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x_hidden.append(x)
                    x, r = self.conv2(g, x, r, self.edge_type, self.edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x_hidden.append(x)
                    x, r = self.conv3(g, x, r, self.edge_type, self.edge_norm)
                    x = drop2(x)
                    x_hidden.append(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(g, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    g, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        x_hidden = torch.stack(x_hidden, dim=1)

        return x_hidden, x

    def forward_base_rel_subgraph(self, bg, subj, obj, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed[bg.ndata[NID]], self.init_rel  # embedding of relations
        edge_type = self.edge_type[bg.edata[EID]]
        edge_norm = self.edge_norm[bg.edata[EID]]
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        bg, x, r, edge_type, edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(bg, x, r, edge_type, edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(bg, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    bg, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        bg.ndata['h'] = x
        sub_list = []
        obj_list = []
        for idx, g in enumerate(dgl.unbatch(bg)):
            head_idx = torch.where(g.ndata[NID] == subj[idx])[0]
            tail_idx = torch.where(g.ndata[NID] == obj[idx])[0]
            head_emb = g.ndata['h'][head_idx]
            tail_emb = g.ndata['h'][tail_idx]
            sub_list.append(head_emb)
            obj_list.append(tail_emb)
        sub_emb = torch.cat(sub_list, dim=0)
        obj_emb = torch.cat(obj_list, dim=0)
        # # filter out embeddings of subjects in this batch
        # sub_emb = torch.index_select(x, 0, subj)
        # # filter out embeddings of objects in this batch
        # obj_emb = torch.index_select(x, 0, obj)

        return sub_emb, obj_emb, x, r

    def forward_base_rel_subgraph_trans(self, bg, input_ids, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed[bg.ndata[NID]], self.init_rel  # embedding of relations
        edge_type = self.edge_type[bg.edata[EID]]
        edge_norm = self.edge_norm[bg.edata[EID]]
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        bg, x, r, edge_type, edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(bg, x, r, edge_type, edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(bg, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    bg, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        bg.ndata['h'] = x
        input_emb = self.init_embed[input_ids]
        for idx, g in enumerate(dgl.unbatch(bg)):
            # print(g.ndata['h'].size())
            # print(input_emb[idx][:].size())
            input_emb[idx][1:g.num_nodes() + 1] = g.ndata['h']
        # sub_list = []
        # obj_list = []
        # for idx, g in enumerate(dgl.unbatch(bg)):
        #     head_idx = torch.where(g.ndata[NID] == subj[idx])[0]
        #     tail_idx = torch.where(g.ndata[NID] == obj[idx])[0]
        #     head_emb = g.ndata['h'][head_idx]
        #     tail_emb = g.ndata['h'][tail_idx]
        #     sub_list.append(head_emb)
        #     obj_list.append(tail_emb)
        # sub_emb = torch.cat(sub_list,dim=0)
        # obj_emb = torch.cat(obj_list,dim=0)
        # # filter out embeddings of subjects in this batch
        # sub_emb = torch.index_select(x, 0, subj)
        # # filter out embeddings of objects in this batch
        # obj_emb = torch.index_select(x, 0, obj)

        return input_emb

    def forward_base_rel_subgraph_trans_new(self, bg, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed[bg.ndata[NID]], self.init_rel  # embedding of relations
        # print(bg.ndata[NID])
        # print(self.edge_type.size())
        # exit(0)
        edge_type = self.edge_type[bg.edata[EID]]
        edge_norm = self.edge_norm[bg.edata[EID]]
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    # print(bg)
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        bg, x, r, edge_type, edge_norm) if self.n_layer == 2 else (x, r)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(bg, x, r, edge_type, edge_norm)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(bg, x, r, edge_type, edge_norm)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(bg, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    bg, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        bg.ndata['h'] = x

        return bg, r

    def forward_base_no_transform(self, subj, rel):
        x, r = self.init_embed, self.init_rel
        sub_emb = torch.index_select(x, 0, subj)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x

    def forward_base_subgraph_search(self, bg, drop1, drop2):
        """
        :param g: graph
        :param sub: subjects in a batch [batch]
        :param rel: relations in a batch [batch]
        :param drop1: dropout rate in first layer
        :param drop2: dropout rate in second layer
        :return: sub_emb: [batch, D]
                 rel_emb: [num_rel*2, D]
                 x: [num_ent, D]
        """
        x, r = self.init_embed[bg.ndata[NID]], self.init_rel  # embedding of relations
        edge_type = self.edge_type[bg.edata[EID]]
        edge_norm = self.edge_norm[bg.edata[EID]]
        x_hidden = []
        if self.n_layer > 0:
            if self.encoder == 'compgcn':
                if self.n_layer < 3:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(
                        bg, x, r, edge_type, edge_norm) if self.n_layer == 2 else (x, r)
                    x_hidden.append(x)
                    x = drop2(x) if self.n_layer == 2 else x
                else:
                    x, r = self.conv1(bg, x, r, edge_type, edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv2(bg, x, r, edge_type, edge_norm)
                    x_hidden.append(x)
                    x = drop1(x)  # embeddings of entities [num_ent, dim]
                    x, r = self.conv3(bg, x, r, edge_type, edge_norm)
                    x_hidden.append(x)
                    x = drop2(x)
            elif self.encoder == 'rgcn':
                x = self.conv1(bg, x, self.edge_type,
                               self.edge_norm.unsqueeze(-1))
                x = drop1(x)  # embeddings of entities [num_ent, dim]
                x = self.conv2(
                    bg, x, self.edge_type, self.edge_norm.unsqueeze(-1)) if self.n_layer == 2 else x
                x = drop2(x) if self.n_layer == 2 else x

        x_hidden = torch.stack(x_hidden, dim=1)

        return x_hidden



class GCN_TransE(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., gamma=9., wni=False, wsi=False, encoder='compgcn',
                 use_bn=True, ltr=True):
        super(GCN_TransE, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                         edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.drop = nn.Dropout(hid_drop)
        self.gamma = gamma

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)

        score = torch.sigmoid(x)

        return score


class GCN_DistMult(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True):
        super(GCN_DistMult, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                           edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.drop = nn.Dropout(hid_drop)

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.drop)
        obj_emb = sub_emb * rel_emb  # [batch_size, emb_dim]
        x = torch.mm(obj_emb, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score


class GCN_ConvE(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(GCN_ConvE, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                        edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        # one channel, do bn on initial embedding
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(
            self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(
            self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(
            self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(
            self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        # fully connected projection
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        self.cat_type = 'multii'

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, rel_embed], 1)

        assert self.embed_dim == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)
        return stack_input

    def forward(self, g, subj, rel):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_emb, rel_emb, all_ent = self.forward_base(
            g, subj, rel, self.drop, self.input_drop)
        # [batch_size, 1, 2*k_h, k_w]
        stack_input = self.concat(sub_emb, rel_emb)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def forward_search(self, g, subj, rel):
        sub_emb, rel_emb, all_ent, hidden_all_ent = self.forward_base_search(
            g, subj, rel, self.drop, self.input_drop)

        return hidden_all_ent

    def compute_pred(self, hidden_x, subj, obj):
        # raise NotImplementedError
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        print(h.size())

    def cross_pair(self, x_i, x_all):
        x = []
        # print(x_i.size())
        # print(x_all.size())
        x_all = x_all.repeat(x_i.size(0), 1, 1, 1)
        # print(x_all.size())
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                if self.cat_type == 'multi':
                    # print(x_i[:, i, :].size())
                    # print(x_all[:,:,j,:].size())
                    test = x_i[:, i, :].unsqueeze(1) * x_all[:, :, j, :]
                    # print(test.size())
                    x.append(test)
                else:
                    test = torch.cat([x_i[:, i, :].unsqueeze(1), x_all[:, :, j, :]], dim=1)
                    print(test.size())
                    x.append(test)
        x = torch.stack(x, dim=1)
        return x


class GCN_ConvE_Rel(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True, input_type='subgraph'):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(GCN_ConvE_Rel, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                            edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr,
                                            input_type)
        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h
        self.n_layer = n_layer

        # one channel, do bn on initial embedding
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(
            self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(
            self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(
            self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(
            self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        # fully connected projection
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)
        self.combine_type = 'concat'

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, rel_embed], 1)

        assert self.embed_dim == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)
        return stack_input

    def forward(self, g, subj, obj):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        if self.input_type == 'subgraph':
            sub_emb, obj_emb, all_ent, all_rel = self.forward_base_rel_subgraph(g, subj, obj, self.drop,
                                                                                self.input_drop)
        else:
            sub_emb, obj_emb, all_ent, all_rel = self.forward_base_rel(g, subj, obj, self.drop, self.input_drop)
        # [batch_size, 1, 2*k_h, k_w]
        stack_input = self.concat(sub_emb, obj_emb)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias_rel.expand_as(x)
        # score = torch.sigmoid(x)
        score = x
        return score

    def forward_search(self, g, subj, obj):
        hidden_all_ent, all_rel = self.forward_base_rel_search(
            g, subj, obj, self.drop, self.input_drop)

        return hidden_all_ent, all_rel

    def compute_pred(self, hidden_x, all_rel, subj, obj, subgraph_sampler, mode='search'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = subgraph_sampler(h, mode)
        # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
        # print(subgraph_sampler(h,mode='argmax'))
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        h = torch.sum(h, dim=1)
        # print(h.size())  # [batch_size, 2*dim]
        h = h.reshape(-1, 1, 2 * self.k_h, self.k_w)
        x = self.bn0(h)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias_rel.expand_as(x)
        # score = torch.sigmoid(x)
        score = x
        return score
        # print(h.size())

    def compute_pred_rs(self, hidden_x, all_rel, subj, obj, random_hops):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = torch.zeros(h.size(0), h.size(1)).to('cuda:0')
        for i in range(h.size(0)):
            atten_matrix[i][self.n_layer * (random_hops[i][0] - 1) + random_hops[i][1] - 1] = 1
        # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
        # print(subgraph_sampler(h,mode='argmax'))
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        h = torch.sum(h, dim=1)
        # print(h.size())  # [batch_size, 2*dim]
        h = h.reshape(-1, 1, 2 * self.k_h, self.k_w)
        x = self.bn0(h)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias_rel.expand_as(x)
        # score = torch.sigmoid(x)
        score = x
        return score
        # print(h.size())

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                if self.combine_type == 'multi':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                elif self.combine_type == 'concat':
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def vis_hop_distribution(self, hidden_x, all_rel, subj, obj, subgraph_sampler, mode='search'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        atten_matrix = subgraph_sampler(h, mode)
        return torch.sum(atten_matrix, dim=0)

    def vis_hop_distribution_rs(self, hidden_x, all_rel, subj, obj, random_hops):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = torch.zeros(h.size(0), h.size(1)).to('cuda:0')
        for i in range(h.size(0)):
            atten_matrix[i][self.n_layer * (random_hops[i][0] - 1) + random_hops[i][1] - 1] = 1
        return torch.sum(atten_matrix, dim=0)


class GCN_Transformer(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True, input_type='subgraph',
                 d_model=100, num_transformer_layers=2, nhead=8, dim_feedforward=100, transformer_dropout=0.1,
                 transformer_activation='relu',
                 graph_pooling='cls', concat_type="gso", max_input_len=100, loss_type='ce'):
        super(GCN_Transformer, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                              edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr,
                                              input_type, loss_type)

        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.num_filt = num_filt
        self.ker_sz, self.k_w, self.k_h = ker_sz, k_w, k_h

        # one channel, do bn on initial embedding
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(
            self.num_filt)  # do bn on output of conv
        self.bn2 = torch.nn.BatchNorm1d(self.embed_dim)

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(
            self.input_drop)  # stacked input dropout
        self.feature_drop = torch.nn.Dropout(
            self.feat_drop)  # feature map dropout
        self.hidden_drop = torch.nn.Dropout(
            self.conve_hid_drop)  # hidden layer dropout

        self.conv2d = torch.nn.Conv2d(in_channels=1, out_channels=self.num_filt,
                                      kernel_size=(self.ker_sz, self.ker_sz), stride=1, padding=0, bias=bias)

        flat_sz_h = int(2 * self.k_h) - self.ker_sz + 1  # height after conv
        flat_sz_w = self.k_w - self.ker_sz + 1  # width after conv
        self.flat_sz = flat_sz_h * flat_sz_w * self.num_filt
        # fully connected projection
        self.fc = torch.nn.Linear(self.flat_sz, self.embed_dim)

        self.d_model = d_model
        self.num_layer = num_transformer_layers
        self.gnn2transformer = nn.Linear(gcn_dim, d_model)
        # Creating Transformer Encoder Model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, transformer_dropout, transformer_activation
        )
        encoder_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers, encoder_norm)
        # self.max_input_len = args.max_input_len
        self.norm_input = nn.LayerNorm(d_model)
        self.graph_pooling = graph_pooling
        self.max_input_len = max_input_len
        self.concat_type = concat_type
        self.cls_embedding = None
        if self.graph_pooling == "cls":
            self.cls_embedding = nn.Parameter(torch.randn([1, 1, self.d_model], requires_grad=True))
        if self.concat_type == "gso":
            self.fc1 = nn.Linear(d_model * 3, 256)
        elif self.concat_type == "so":
            self.fc1 = nn.Linear(d_model * 2, 256)
        elif self.concat_type == "g":
            self.fc1 = nn.Linear(d_model, 256)
        self.fc2 = nn.Linear(256, num_rel * 2)

    def forward(self, bg, input_ids, subj, obj):
        # tokens_emb = self.forward_base_rel_subgraph_trans(bg,input_ids,self.drop,self.input_drop)
        # tokens_emb = tokens_emb.permute(1,0,2) # (s, b, d)
        # tokens_emb = self.gnn2transformer(tokens_emb)
        # padding_mask = get_pad_mask(input_ids, self.num_ent+1)
        # tokens_emb = self.norm_input(tokens_emb)
        # transformer_out = self.transformer(tokens_emb, src_key_padding_mask=padding_mask)
        # h_graph = transformer_out[0]
        # output = self.fc_out(h_graph)
        # output = torch.sigmoid(output)
        # return output

        bg, all_rel = self.forward_base_rel_subgraph_trans_new(bg, self.drop, self.input_drop)
        batch_size = bg.batch_size
        h_node = self.gnn2transformer(bg.ndata['h'])
        padded_h_node, src_padding_mask, subj_idx, obj_idx = pad_batch(h_node, bg, self.max_input_len, subj, obj)
        if self.cls_embedding is not None:
            expand_cls_embedding = self.cls_embedding.expand(1, padded_h_node.size(1), -1)
            padded_h_node = torch.cat([padded_h_node, expand_cls_embedding], dim=0)
            zeros = src_padding_mask.data.new(src_padding_mask.size(0), 1).fill_(0)
            src_padding_mask = torch.cat([src_padding_mask, zeros], dim=1)
        padded_h_node = self.norm_input(padded_h_node)
        transformer_out = self.transformer(padded_h_node, src_key_padding_mask=src_padding_mask)
        subj_emb_list = []
        obj_emb_list = []
        for batch_idx in range(batch_size):
            subj_emb = transformer_out[subj_idx[batch_idx], batch_idx, :]
            obj_emb = transformer_out[obj_idx[batch_idx], batch_idx, :]
            subj_emb_list.append(subj_emb)
            obj_emb_list.append(obj_emb)
        subj_emb = torch.stack(subj_emb_list)
        obj_emb = torch.stack(obj_emb_list)
        if self.graph_pooling in ["last", "cls"]:
            h_graph = transformer_out[-1]
            if self.concat_type == "gso":
                h_repr = torch.cat([h_graph, subj_emb, obj_emb], dim=1)
            elif self.concat_type == "g":
                h_repr = h_graph
        else:
            if self.concat_type == "so":
                h_repr = torch.cat([subj_emb, obj_emb], dim=1)
        h_repr = F.relu(self.fc1(h_repr))
        score = self.fc2(h_repr)
        # print(score.size())
        # print(score)
        # score = self.conve(subj_emb, obj_emb, all_rel)

        return score

    def conve_rel(self, sub_emb, obj_emb, all_rel):
        stack_input = self.concat(sub_emb, obj_emb)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias_rel.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def concat(self, ent_embed, rel_embed):
        """
        :param ent_embed: [batch_size, embed_dim]
        :param rel_embed: [batch_size, embed_dim]
        :return: stack_input: [B, C, H, W]
        """
        ent_embed = ent_embed.view(-1, 1, self.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.embed_dim)
        # [batch_size, 2, embed_dim]
        stack_input = torch.cat([ent_embed, rel_embed], 1)

        assert self.embed_dim == self.k_h * self.k_w
        # reshape to 2D [batch, 1, 2*k_h, k_w]
        stack_input = stack_input.reshape(-1, 1, 2 * self.k_h, self.k_w)
        return stack_input

    def conve_ent(self, sub_emb, rel_emb, all_ent):
        stack_input = self.concat(sub_emb, rel_emb)
        x = self.bn0(stack_input)
        x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        x = self.fc(x)  # [batch_size, embed_dim]
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, all_ent.transpose(1, 0))  # [batch_size, ent_num]
        x += self.bias.expand_as(x)
        score = torch.sigmoid(x)
        return score

    def evaluate(self, subj, obj):
        sub_emb, rel_emb, all_ent = self.forward_base_no_transform(subj, obj)
        score = self.conve_ent(sub_emb, rel_emb, all_ent)
        return score


class GCN_None(GCNs):

    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True,
                 input_type='subgraph', graph_pooling='mean', concat_type='gso', loss_type='ce', add_reverse=True):
        super(GCN_None, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                       edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder, use_bn, ltr,
                                       input_type, loss_type, add_reverse)

        self.hid_drop, self.input_drop, self.conve_hid_drop, self.feat_drop = hid_drop, input_drop, conve_hid_drop, feat_drop
        self.graph_pooling_type = graph_pooling
        self.concat_type = concat_type
        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(
            self.input_drop)  # stacked input dropout
        if self.concat_type == "gso":
            self.fc1 = nn.Linear(gcn_dim * 3, 256)
        elif self.concat_type == "so":
            self.fc1 = nn.Linear(gcn_dim * 2, 256)
        elif self.concat_type == "g":
            self.fc1 = nn.Linear(gcn_dim, 256)
        if add_reverse:
            self.fc2 = nn.Linear(256, num_rel * 2)
        else:
            self.fc2 = nn.Linear(256, num_rel)

    def forward(self, g, subj, obj):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        # start_time = time.time()
        bg, all_rel = self.forward_base_rel_subgraph_trans_new(g, self.drop, self.input_drop)
        # print(f'{time.time() - start_time:.2f}s')
        h_repr_list = []
        for batch_idx, g in enumerate(dgl.unbatch(bg)):
            h_graph = readout_nodes(g, 'h', op=self.graph_pooling_type)
            sub_idx = torch.where(g.ndata[NID] == subj[batch_idx])[0]
            obj_idx = torch.where(g.ndata[NID] == obj[batch_idx])[0]
            sub_emb = g.ndata['h'][sub_idx]
            obj_emb = g.ndata['h'][obj_idx]
            h_repr = torch.cat([h_graph, sub_emb, obj_emb], dim=1)
            h_repr_list.append(h_repr)
        h_repr = torch.stack(h_repr_list, dim=0).squeeze(1)
        # print(f'{time.time() - start_time:.2f}s')
        score = F.relu(self.fc1(h_repr))
        score = self.fc2(score)

        return score


class GCN_MLP(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True, input_type='subgraph', graph_pooling='mean', combine_type='mult', loss_type='ce',
                 add_reverse=True):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(GCN_MLP, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                      edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder,
                                      use_bn, ltr, input_type, loss_type, add_reverse)
        self.hid_drop, self.input_drop = hid_drop, input_drop
        if add_reverse:
            self.num_rel = num_rel * 2
        else:
            self.num_rel = num_rel

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout

        self.combine_type = combine_type
        if self.combine_type == 'concat':
            self.fc = torch.nn.Linear(2 * self.embed_dim, self.num_rel)
        elif self.combine_type == 'mult':
            self.fc = torch.nn.Linear(self.embed_dim, self.num_rel)

    def forward(self, g, subj, obj):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        if self.input_type == 'subgraph':
            print('1')
            # sub_emb, obj_emb, all_ent, all_rel = self.forward_base_rel_subgraph(g, subj, obj, self.drop, self.input_drop)
            bg, all_rel = self.forward_base_rel_subgraph_trans_new(g, self.drop, self.input_drop)
            sub_ids = (bg.ndata['id'] == 1).nonzero().squeeze(1)
            sub_embs = bg.ndata['h'][sub_ids]
            obj_ids = (bg.ndata['id'] == 2).nonzero().squeeze(1)
            obj_embs = bg.ndata['h'][obj_ids]
            h_graph = readout_nodes(bg, 'h', op=self.graph_pooling_type)
            # edge_embs = torch.concat([sub_embs, obj_embs], dim=1)
            # score = self.fc(edge_embs)
        else:
            sub_embs, obj_embs, all_ent, all_rel = self.forward_base_rel(g, subj, obj, self.drop, self.input_drop)
        if self.combine_type == 'concat':
            edge_embs = torch.concat([sub_embs, obj_embs], dim=1)
        elif self.combine_type == 'mult':
            edge_embs = sub_embs * obj_embs
        else:
            raise NotImplementedError
        score = self.fc(edge_embs)
        return score

    def compute_vid_pred(self, hidden_x, subj, obj):
        scores = []
        scores_sigmoid = []
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        for i in range(h.size()[1]):
        # print(h.size())
        # exit(0)
            edge_embs = h[:,i,:]
        # if self.combine_type == 'concat':
        #     edge_embs = torch.concat([sub_embs, obj_embs], dim=1)
        # elif self.combine_type == 'mult':
        #     edge_embs = sub_embs * obj_embs
        # else:
        #     raise NotImplementedError
        #     print(edge_embs.size())
            score = self.fc(edge_embs)
            scores.append(score)
            # score_sigmoid = torch.sigmoid(score)
            # print(score.size())
            # scores.append(torch.max(score[:6,:],1))
            # scores_sigmoid.append(torch.max(score_sigmoid[:6,:],1))
            # print(scores[-1])
        return scores

    def forward_search(self, g, mode='allgraph'):
        # if mode == 'allgraph':
        hidden_all_ent, all_ent = self.forward_base_rel_search(
            g, self.drop, self.input_drop)
        # elif mode == 'subgraph':
        #     hidden_all_ent = self.forward_base_subgraph_search(
        #         g, self.drop, self.input_drop)

        return hidden_all_ent, all_ent

    def compute_pred(self, hidden_x, subj, obj, subgraph_sampler, mode='search', search_algorithm='darts'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = subgraph_sampler(h, mode, search_algorithm)
        # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
        # print(subgraph_sampler(h,mode='argmax'))
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        h = torch.sum(h, dim=1)
        # print(h.size())  # [batch_size, 2*dim]
        score = self.fc(h)
        return score

    def compute_mix_hop_pred(self, hidden_x, subj, obj, hop_index):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        edge_embs = h[:,hop_index,:]
        score = self.fc(edge_embs)
        return score
    # def fine_tune_with_implicit_subgraph(self, all_ent, subgraph, subj, obj):
    #     sg_list = []
    #     for idx in range(subgraph.size(0)):
    #         sg_list.append(torch.mean(all_ent[subgraph[idx,:]], dim=0).unsqueeze(0))
    #     sg_embs = torch.concat(sg_list)
    #     # print(sg_embs.size())
    #     sub_embs = torch.index_select(all_ent, 0, subj)
    #     # print(sub_embs.size())
    #     # filter out embeddings of relations in this batch
    #     obj_embs = torch.index_select(all_ent, 0, obj)
    #     # print(obj_embs.size())
    #     edge_embs = torch.concat([sub_embs, obj_embs, sg_embs], dim=1)
    #     score = self.predictor(edge_embs)
    #     # print(F.embedding(subgraph, all_ent))
    #     return score

    # def compute_pred_rs(self, hidden_x, all_rel, subj, obj, random_hops):
    #     h = self.cross_pair(hidden_x[subj], hidden_x[obj])
    #     # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
    #     atten_matrix = torch.zeros(h.size(0),h.size(1)).to('cuda:0')
    #     for i in range(h.size(0)):
    #         atten_matrix[i][self.n_layer*(random_hops[i][0]-1)+random_hops[i][1]-1] = 1
    #     # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
    #     # print(subgraph_sampler(h,mode='argmax'))
    #     n, c = atten_matrix.shape
    #     h = h * atten_matrix.view(n,c,1)
    #     # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
    #     h = torch.sum(h,dim=1)
    #     # print(h.size())  # [batch_size, 2*dim]
    #     h = h.reshape(-1, 1, 2 * self.k_h, self.k_w)
    #     # x = self.bn0(h)
    #     # x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
    #     # x = self.bn1(x)
    #     # x = F.relu(x)
    #     # x = self.feature_drop(x)
    #     # x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
    #     # x = self.fc(x)  # [batch_size, embed_dim]
    #     # x = self.hidden_drop(x)
    #     # x = self.bn2(x)
    #     # x = F.relu(x)
    #     # x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
    #     # x += self.bias_rel.expand_as(x)
    #     # # score = torch.sigmoid(x)
    #     # score = x
    #     return score
    # print(h.size())

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                if self.combine_type == 'mult':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                elif self.combine_type == 'concat':
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def vis_hop_distribution(self, hidden_x, subj, obj, subgraph_sampler, mode='search'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        atten_matrix = subgraph_sampler(h, mode)
        return torch.sum(atten_matrix, dim=0)

    def vis_hop_distribution_rs(self, hidden_x, subj, obj, random_hops):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = torch.zeros(h.size(0), h.size(1)).to('cuda:0')
        for i in range(h.size(0)):
            atten_matrix[i][self.n_layer * (random_hops[i][0] - 1) + random_hops[i][1] - 1] = 1
        return torch.sum(atten_matrix, dim=0)


class GCN_MLP_NCN(GCNs):
    def __init__(self, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., opn='mult', hid_drop=0., input_drop=0., conve_hid_drop=0., feat_drop=0.,
                 num_filt=None, ker_sz=None, k_h=None, k_w=None, wni=False, wsi=False, encoder='compgcn', use_bn=True,
                 ltr=True, input_type='subgraph', graph_pooling='mean', combine_type='mult', loss_type='ce',
                 add_reverse=True):
        """
        :param num_ent: number of entities
        :param num_rel: number of different relations
        :param num_base: number of bases to use
        :param init_dim: initial dimension
        :param gcn_dim: dimension after first layer
        :param embed_dim: dimension after second layer
        :param n_layer: number of layer
        :param edge_type: relation type of each edge, [E]
        :param bias: weather to add bias
        :param gcn_drop: dropout rate in compgcncov
        :param opn: combination operator
        :param hid_drop: gcn output (embedding of each entity) dropout
        :param input_drop: dropout in conve input
        :param conve_hid_drop: dropout in conve hidden layer
        :param feat_drop: feature dropout in conve
        :param num_filt: number of filters in conv2d
        :param ker_sz: kernel size in conv2d
        :param k_h: height of 2D reshape
        :param k_w: width of 2D reshape
        """
        super(GCN_MLP_NCN, self).__init__(num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                          edge_type, edge_norm, bias, gcn_drop, opn, wni, wsi, encoder,
                                          use_bn, ltr, input_type, loss_type, add_reverse)
        self.hid_drop, self.input_drop = hid_drop, input_drop
        self.num_rel = num_rel

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout

        # fully connected projection
        self.combine_type = combine_type
        self.graph_pooling_type = graph_pooling
        self.lin_layers = 2
        self._init_predictor()

    def _init_predictor(self):
        self.lins = torch.nn.ModuleList()
        if self.combine_type == 'mult':
            input_channels = self.embed_dim
        else:
            input_channels = self.embed_dim * 2
        self.lins.append(torch.nn.Linear(input_channels + self.embed_dim, self.embed_dim))
        for _ in range(self.lin_layers - 2):
            self.lins.append(torch.nn.Linear(self.embed_dim, self.embed_dim))
        self.lins.append(torch.nn.Linear(self.embed_dim, self.num_rel))

    def forward(self, g, subj, obj, cns):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """
        sub_embs, obj_embs, all_ent, all_rel = self.forward_base_rel(g, subj, obj, self.drop, self.input_drop)
        cn_embs = self.get_common_1hopneighbor_emb(all_ent, cns)

        if self.combine_type == 'mult':
            edge_embs = torch.concat([sub_embs * obj_embs, cn_embs], dim=1)
        elif self.combine_type == 'concat':
            edge_embs = torch.concat([sub_embs, obj_embs, cn_embs], dim=1)
        x = edge_embs
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        score = self.lins[-1](x)

        # [batch_size, 1, 2*k_h, k_w]
        # stack_input = self.concat(sub_emb, obj_emb)
        # x = self.bn0(stack_input)
        # x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = self.feature_drop(x)
        # x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
        # x = self.fc(x)  # [batch_size, embed_dim]
        # x = self.hidden_drop(x)
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
        # x += self.bias_rel.expand_as(x)
        # # score = torch.sigmoid(x)
        # score = x
        return score

    def forward_search(self, g, mode='allgraph'):
        hidden_all_ent, all_ent = self.forward_base_rel_search(
            g, self.drop, self.input_drop)

        return hidden_all_ent, all_ent

    def compute_pred(self, hidden_x, all_ent, subj, obj, cns, subgraph_sampler, mode='search'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = subgraph_sampler(h, mode)
        # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
        # print(subgraph_sampler(h,mode='argmax'))
        n, c = atten_matrix.shape
        h = h * atten_matrix.view(n, c, 1)
        cn_embs = self.get_common_1hopneighbor_emb(all_ent, cns)
        # cn = self.get_common_neighbor_emb(hidden_x, cns)
        # cn = self.get_common_neighbor_emb_(all_ent, cns)
        # cn = cn * atten_matrix.view(n, c, 1)

        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        h = torch.sum(h, dim=1)
        # cn = torch.sum(cn, dim=1)
        concat_embs = torch.concat([h, cn_embs], dim=1)
        x = concat_embs
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        score = self.lins[-1](x)
        # score = self.fc(h)
        # if self.combine_type == 'so':
        #     score = self.fc(h)
        # elif self.combine_type == 'gso':
        #     bg, all_rel = self.forward_base_rel_subgraph_trans_new(g, self.drop, self.input_drop)
        #     h_graph = readout_nodes(bg, 'h', op=self.graph_pooling_type)
        #     edge_embs = torch.concat([h_graph, h], dim=1)
        #     score = self.predictor(edge_embs)
        # print(h.size())  # [batch_size, 2*dim]

        return score

    #     # print(h.size())

    def get_common_neighbor_emb(self, hidden_x, cns):
        x = []
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                x_tmp = []
                for idx in range(cns.size(0)):
                    print(hidden_x[cns[idx, i * 2 + j, :], i, :].size())
                    print(hidden_x[cns[idx, i * 2 + j, :], j, :].size())
                    x_tmp.append(torch.mean(hidden_x[cns[idx, i * 2 + j, :], i * 2 + j, :], dim=0))
                x.append(torch.stack(x_tmp, dim=0))
        x = torch.stack(x, dim=1)
        return x

    def get_common_neighbor_emb_(self, all_ent, cns):
        x = []
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                x_tmp = []
                for idx in range(cns.size(0)):
                    x_tmp.append(torch.mean(all_ent[cns[idx, i * 2 + j, :]], dim=0))
                x.append(torch.stack(x_tmp, dim=0))
        x = torch.stack(x, dim=1)
        return x

    def get_common_1hopneighbor_emb(self, all_ent, cns):
        x = []
        for idx in range(cns.size(0)):
            x.append(torch.mean(all_ent[cns[idx, :], :], dim=0))
        cn_embs = torch.stack(x, dim=0)
        return cn_embs

    # def compute_pred_rs(self, hidden_x, all_rel, subj, obj, random_hops):
    #     h = self.cross_pair(hidden_x[subj], hidden_x[obj])
    #     # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
    #     atten_matrix = torch.zeros(h.size(0),h.size(1)).to('cuda:0')
    #     for i in range(h.size(0)):
    #         atten_matrix[i][self.n_layer*(random_hops[i][0]-1)+random_hops[i][1]-1] = 1
    #     # print(atten_matrix.size()) # [batch_size, encoder_layer^2]
    #     # print(subgraph_sampler(h,mode='argmax'))
    #     n, c = atten_matrix.shape
    #     h = h * atten_matrix.view(n,c,1)
    #     # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
    #     h = torch.sum(h,dim=1)
    #     # print(h.size())  # [batch_size, 2*dim]
    #     h = h.reshape(-1, 1, 2 * self.k_h, self.k_w)
    #     # x = self.bn0(h)
    #     # x = self.conv2d(x)  # [batch_size, num_filt, flat_sz_h, flat_sz_w]
    #     # x = self.bn1(x)
    #     # x = F.relu(x)
    #     # x = self.feature_drop(x)
    #     # x = x.view(-1, self.flat_sz)  # [batch_size, flat_sz]
    #     # x = self.fc(x)  # [batch_size, embed_dim]
    #     # x = self.hidden_drop(x)
    #     # x = self.bn2(x)
    #     # x = F.relu(x)
    #     # x = torch.mm(x, all_rel.transpose(1, 0))  # [batch_size, ent_num]
    #     # x += self.bias_rel.expand_as(x)
    #     # # score = torch.sigmoid(x)
    #     # score = x
    #     return score
    # print(h.size())

    def cross_pair(self, x_i, x_j):
        x = []
        for i in range(self.n_layer):
            for j in range(self.n_layer):
                if self.combine_type == 'mult':
                    x.append(x_i[:, i, :] * x_j[:, j, :])
                elif self.combine_type == 'concat':
                    x.append(torch.cat([x_i[:, i, :], x_j[:, j, :]], dim=1))
        x = torch.stack(x, dim=1)
        return x

    def vis_hop_distribution(self, hidden_x, subj, obj, subgraph_sampler, mode='search'):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        atten_matrix = subgraph_sampler(h, mode)
        return torch.sum(atten_matrix, dim=0)

    def vis_hop_distribution_rs(self, hidden_x, subj, obj, random_hops):
        h = self.cross_pair(hidden_x[subj], hidden_x[obj])
        # print(h.size()) # [batch_size, encoder_layer^2, 2*dim]
        atten_matrix = torch.zeros(h.size(0), h.size(1)).to('cuda:0')
        for i in range(h.size(0)):
            atten_matrix[i][self.n_layer * (random_hops[i][0] - 1) + random_hops[i][1] - 1] = 1
        return torch.sum(atten_matrix, dim=0)


def get_pad_mask(seq, pad_idx):
    return (seq == pad_idx)


def pad_batch(h_node, bg, max_input_len, subj, obj):
    subj_list = []
    obj_list = []
    batch_size = bg.batch_size
    max_num_nodes = min(max(bg.batch_num_nodes()).item(), max_input_len)
    padded_h_node = h_node.data.new(max_num_nodes, batch_size, h_node.size(-1)).fill_(0)
    src_padding_mask = h_node.data.new(batch_size, max_num_nodes).fill_(0).bool()
    for batch_idx, g in enumerate(dgl.unbatch(bg)):
        num_nodes = g.num_nodes()
        padded_h_node[-num_nodes:, batch_idx] = g.ndata['h']
        src_padding_mask[batch_idx, : max_num_nodes - num_nodes] = True
        subj_idx = torch.where(g.ndata[NID] == subj[batch_idx])[0] + max_num_nodes - num_nodes
        obj_idx = torch.where(g.ndata[NID] == obj[batch_idx])[0] + max_num_nodes - num_nodes
        subj_list.append(subj_idx)
        obj_list.append(obj_idx)
    subj_idx = torch.cat(subj_list)
    obj_idx = torch.cat(obj_list)
    return padded_h_node, src_padding_mask, subj_idx, obj_idx
