import torch
from torch import nn
import dgl
import dgl.function as fn
from torch.autograd import Variable
from model.genotypes import COMP_PRIMITIVES, AGG_PRIMITIVES, COMB_PRIMITIVES, ACT_PRIMITIVES
from model.operations import *
from model.search_layer import SearchGCNConv
from torch.nn.functional import softmax
from pprint import pprint


class CompMixOP(nn.Module):
    def __init__(self):
        super(CompMixOP, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in COMP_PRIMITIVES:
            op = COMP_OPS[primitive]()
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, src_emb, rel_emb, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(src_emb, rel_emb))
        return sum(mixed_res)


class CompOp(nn.Module):

    def __init__(self, op_name):
        super(CompOp, self).__init__()
        self.op = COMP_OPS[op_name]()

    def reset_parameters(self):
        self.op.reset_parameters()

    def forward(self, src_emb, rel_emb):
        return self.op(src_emb, rel_emb)


class Network(nn.Module):
    def __init__(self, args, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 conv_bias=True, gcn_drop=0., wni=False, wsi=False, use_bn=True, ltr=True, loss_type='ce'):
        super(Network, self).__init__()
        self.act = torch.tanh
        self.args = args
        self.loss_type = loss_type
        self.num_ent, self.num_rel, self.num_base = num_ent, num_rel, num_base
        self.init_dim, self.gcn_dim, self.embed_dim = init_dim, gcn_dim, embed_dim

        self.gcn_drop = gcn_drop
        self.edge_type = edge_type  # [E]
        self.edge_norm = edge_norm  # [E]
        self.n_layer = n_layer

        self.init_embed = self.get_param([self.num_ent + 1, self.init_dim])
        self.init_rel = self.get_param([self.num_rel, self.init_dim])
        self.bias_rel = nn.Parameter(torch.zeros(self.num_rel))

        self._initialize_alphas()
        self._initialize_loss()
        self.gnn_layers = nn.ModuleList()
        for idx in range(self.args.n_layer):
            if idx == 0:
                self.gnn_layers.append(
                    SearchGCNConv(self.init_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        comp_weights=self.comp_alphas[idx]))
            elif idx == self.args.n_layer-1:
                self.gnn_layers.append(SearchGCNConv(self.gcn_dim, self.embed_dim, self.act, conv_bias, gcn_drop, num_base=-1,
                              num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                              comp_weights=self.comp_alphas[idx]))
            else:
                self.gnn_layers.append(
                    SearchGCNConv(self.gcn_dim, self.gcn_dim, self.act, conv_bias, gcn_drop, num_base=-1,
                                        num_rel=self.num_rel, wni=wni, wsi=wsi, use_bn=use_bn, ltr=ltr,
                                        comp_weights=self.comp_alphas[idx]))

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):
        gene = []
        comp_max, comp_indices = torch.max(softmax(self.comp_alphas, dim=-1).data.cpu(), dim=-1)
        agg_max, agg_indices = torch.max(softmax(self.agg_alphas, dim=-1).data.cpu(), dim=-1)
        comb_max, comb_indices = torch.max(softmax(self.comb_alphas, dim=-1).data.cpu(), dim=-1)
        act_max, act_indices = torch.max(softmax(self.act_alphas, dim=-1).data.cpu(), dim=-1)
        pprint(comp_max)
        pprint(agg_max)
        pprint(comb_max)
        pprint(act_max)
        for idx in range(self.args.n_layer):
            gene.append(COMP_PRIMITIVES[comp_indices[idx]])
            gene.append(AGG_PRIMITIVES[agg_indices[idx]])
            gene.append(COMB_PRIMITIVES[comb_indices[idx]])
            gene.append(ACT_PRIMITIVES[act_indices[idx]])
        return "||".join(gene)

        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.act = act  # activation function
        # self.device = None
        # if add_reverse:
        #     self.rel = nn.Parameter(torch.empty([num_rel * 2, in_channels], dtype=torch.float))
        # else:
        #     self.rel = nn.Parameter(torch.empty([num_rel, in_channels], dtype=torch.float))
        # self.opn = opn
        #
        # self.use_bn = use_bn
        # self.ltr = ltr
        #
        # # relation-type specific parameter
        # self.in_w = self.get_param([in_channels, out_channels])
        # self.out_w = self.get_param([in_channels, out_channels])
        # self.loop_w = self.get_param([in_channels, out_channels])
        # # transform embedding of relations to next layer
        # self.w_rel = self.get_param([in_channels, out_channels])
        # self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding
        #
        # self.drop = nn.Dropout(drop_rate)
        # self.bn = torch.nn.BatchNorm1d(out_channels)
        # self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # if num_base > 0:
        #     if add_reverse:
        #         self.rel_wt = self.get_param([num_rel * 2, num_base])
        #     else:
        #         self.rel_wt = self.get_param([num_rel, num_base])
        # else:
        #     self.rel_wt = None
        #
        # self.wni = wni
        # self.wsi = wsi

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def _initialize_alphas(self):
        comp_ops_num = len(COMP_PRIMITIVES)
        agg_ops_num = len(AGG_PRIMITIVES)
        comb_ops_num = len(COMB_PRIMITIVES)
        act_ops_num = len(ACT_PRIMITIVES)
        if self.args.search_algorithm == "darts":
            self.comp_alphas = Variable(1e-3 * torch.randn(self.args.n_layer, comp_ops_num).cuda(), requires_grad=True)
            self.agg_alphas = Variable(1e-3 * torch.randn(self.args.n_layer, agg_ops_num).cuda(), requires_grad=True)
            self.comb_alphas = Variable(1e-3 * torch.randn(self.args.n_layer, comb_ops_num).cuda(), requires_grad=True)
            self.act_alphas = Variable(1e-3 * torch.randn(self.args.n_layer, act_ops_num).cuda(), requires_grad=True)
        elif self.args.search_algorithm == "snas":
            # self.comp_alphas = Variable(1e-3 * torch.randn(self.args.n_layer, comp_ops_num).cuda(), requires_grad=True)
            self.comp_alphas = Variable(
                torch.ones(self.args.n_layer, comp_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)
            self.agg_alphas = Variable(
                torch.ones(self.args.n_layer, agg_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)
            self.comb_alphas = Variable(
                torch.ones(self.args.n_layer, comb_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)
            self.act_alphas = Variable(
                torch.ones(self.args.n_layer, act_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
                requires_grad=True)
            # self.la_alphas = Variable(torch.ones(1, la_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
            #                           requires_grad=True)
            # self.seq_alphas = Variable(
            #     torch.ones(self.layer_num, seq_ops_num).normal_(self.args.loc_mean, self.args.loc_std).cuda(),
            #     requires_grad=True)
        else:
            raise NotImplementedError
        self._arch_parameters = [
            self.comp_alphas,
            self.agg_alphas,
            self.comb_alphas,
            self.act_alphas
        ]

    def _initialize_loss(self):
        if self.loss_type == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif self.loss_type == 'bce':
            self.loss = nn.BCELoss(reduce=False)
        elif self.loss_type == 'bce_logits':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError

    def calc_loss(self, pred, label, pos_neg=None):
        if pos_neg is not None:
            m = nn.Sigmoid()
            score_pos = m(pred)
            targets_pos = pos_neg.unsqueeze(1)
            loss = self.loss(score_pos, label * targets_pos)
            return torch.sum(loss * label)
        return self.loss(pred, label)

    def _get_categ_mask(self, alpha):
        # log_alpha = torch.log(alpha)
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.args.temperature)
        return one_hot

    def _get_categ_mask_new(self, alpha):
        # log_alpha = torch.log(alpha)
        print(alpha)
        m = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            torch.tensor([self.args.temperature]).cuda(), alpha)
        print(m.sample())
        print(m.rsample())

    def get_one_hot_alpha(self, alpha):
        one_hot_alpha = torch.zeros_like(alpha, device=alpha.device)
        idx = torch.argmax(alpha, dim=-1)

        for i in range(one_hot_alpha.size(0)):
            one_hot_alpha[i, idx[i]] = 1.0

        return one_hot_alpha

    def forward_base(self, g, subj, obj, drop1, drop2, mode=None):
        if self.args.search_algorithm == "darts":
            comp_weights = softmax(self.comp_alphas / self.args.temperature, dim=-1)
            agg_weights = softmax(self.agg_alphas / self.args.temperature, dim=-1)
            comb_weights = softmax(self.comb_alphas / self.args.temperature, dim=-1)
            act_weights = softmax(self.act_alphas / self.args.temperature, dim=-1)
        elif self.args.search_algorithm == "snas":
            # comp_weights = self._get_categ_mask_new(self.comp_alphas)
            comp_weights = self._get_categ_mask(self.comp_alphas)
            agg_weights = self._get_categ_mask(self.agg_alphas)
            comb_weights = self._get_categ_mask(self.comb_alphas)
            act_weights = self._get_categ_mask(self.act_alphas)
        else:
            raise NotImplementedError
        if mode == 'evaluate_single_path':
            comp_weights = self.get_one_hot_alpha(comp_weights)
            agg_weights = self.get_one_hot_alpha(agg_weights)
            comb_weights = self.get_one_hot_alpha(comb_weights)
            act_weights = self.get_one_hot_alpha(act_weights)
        # weights = dict()
        # weights['comp'] = comp_weights
        x, r = self.init_embed, self.init_rel  # embedding of relations

        for i, layer in enumerate(self.gnn_layers):
            if i != self.args.n_layer-1:
                x, r = layer(g, x, r, self.edge_type, self.edge_norm, comp_weights[i], agg_weights[i], comb_weights[i], act_weights[i])
                x = drop1(x)
            else:
                x, r = layer(g, x, r, self.edge_type, self.edge_norm, comp_weights[i], agg_weights[i], comb_weights[i], act_weights[i])
                x = drop2(x)

        sub_emb = torch.index_select(x, 0, subj)
        # filter out embeddings of objects in this batch
        obj_emb = torch.index_select(x, 0, obj)

        return sub_emb, obj_emb, x, r

    def forward_base_search(self, g, drop1, drop2, mode=None):
        if self.args.search_algorithm == "darts":
            comp_weights = softmax(self.comp_alphas / self.args.temperature, dim=-1)
            agg_weights = softmax(self.agg_alphas / self.args.temperature, dim=-1)
            comb_weights = softmax(self.comb_alphas / self.args.temperature, dim=-1)
            act_weights = softmax(self.act_alphas / self.args.temperature, dim=-1)
        elif self.args.search_algorithm == "snas":
            comp_weights = self._get_categ_mask(self.comp_alphas)
            agg_weights = self._get_categ_mask(self.agg_alphas)
            comb_weights = self._get_categ_mask(self.comb_alphas)
            act_weights = self._get_categ_mask(self.act_alphas)
        else:
            raise NotImplementedError
        if mode == 'evaluate_single_path':
            comp_weights = self.get_one_hot_alpha(comp_weights)
            agg_weights = self.get_one_hot_alpha(agg_weights)
            comb_weights = self.get_one_hot_alpha(comb_weights)
            act_weights = self.get_one_hot_alpha(act_weights)
        x, r = self.init_embed, self.init_rel  # embedding of relations
        x_hidden = []
        for i, layer in enumerate(self.gnn_layers):
            if i != self.args.n_layer-1:
                x, r = layer(g, x, r, self.edge_type, self.edge_norm, comp_weights[i], agg_weights[i], comb_weights[i], act_weights[i])
                x_hidden.append(x)
                x = drop1(x)
            else:
                x, r = layer(g, x, r, self.edge_type, self.edge_norm, comp_weights[i], agg_weights[i], comb_weights[i], act_weights[i])
                x_hidden.append(x)
                x = drop2(x)
        x_hidden = torch.stack(x_hidden, dim=1)

        return x_hidden, x


class SearchGCN_MLP(Network):
    def __init__(self, args, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer, edge_type, edge_norm,
                 bias=True, gcn_drop=0., hid_drop=0., input_drop=0., wni=False, wsi=False, use_bn=True,
                 ltr=True, combine_type='mult', loss_type='ce'):
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
        super(SearchGCN_MLP, self).__init__(args, num_ent, num_rel, num_base, init_dim, gcn_dim, embed_dim, n_layer,
                                      edge_type, edge_norm, bias, gcn_drop, wni, wsi, use_bn, ltr, loss_type)
        self.hid_drop, self.input_drop = hid_drop, input_drop

        self.num_rel = num_rel

        self.drop = torch.nn.Dropout(self.hid_drop)  # gcn output dropout
        self.input_drop = torch.nn.Dropout(self.input_drop)  # stacked input dropout

        self.combine_type = combine_type
        if self.combine_type == 'concat':
            self.fc = torch.nn.Linear(2 * self.embed_dim, self.num_rel)
        elif self.combine_type == 'mult':
            self.fc = torch.nn.Linear(self.embed_dim, self.num_rel)

    def forward(self, g, subj, obj, mode=None):
        """
        :param g: dgl graph
        :param sub: subject in batch [batch_size]
        :param rel: relation in batch [batch_size]
        :return: score: [batch_size, ent_num], the prob in link-prediction
        """

        sub_embs, obj_embs, all_ent, all_rel = self.forward_base(g, subj, obj, self.drop, self.input_drop, mode)
        if self.combine_type == 'concat':
            edge_embs = torch.concat([sub_embs, obj_embs], dim=1)
        elif self.combine_type == 'mult':
            edge_embs = sub_embs * obj_embs
        else:
            raise NotImplementedError
        score = self.fc(edge_embs)
        return score

    def forward_search(self, g, mode=None):
        # if mode == 'allgraph':
        hidden_all_ent, all_ent = self.forward_base_search(
            g, self.drop, self.input_drop, mode)
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
