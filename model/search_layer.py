import torch
from torch import nn
import dgl
import dgl.function as fn
from model.genotypes import COMP_PRIMITIVES, AGG_PRIMITIVES, COMB_PRIMITIVES, ACT_PRIMITIVES
from model.operations import *


class CompMixOp(nn.Module):
    def __init__(self):
        super(CompMixOp, self).__init__()
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


class AggMixOp(nn.Module):
    def __init__(self):
        super(AggMixOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in AGG_PRIMITIVES:
            op = AGG_OPS[primitive]()
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, msg, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(msg))
        return sum(mixed_res)


class CombMixOp(nn.Module):
    def __init__(self, out_channels):
        super(CombMixOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in COMB_PRIMITIVES:
            op = COMB_OPS[primitive](out_channels)
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, self_emb, msg, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(self_emb, msg))
        return sum(mixed_res)


class ActMixOp(nn.Module):
    def __init__(self):
        super(ActMixOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in ACT_PRIMITIVES:
            op = ACT_OPS[primitive]()
            self._ops.append(op)

    def reset_parameters(self):
        for op in self._ops:
            op.reset_parameters()

    def forward(self, emb, weights):
        mixed_res = []
        for w, op in zip(weights, self._ops):
            mixed_res.append(w * op(emb))
        return sum(mixed_res)

class SearchGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, act=lambda x: x, bias=True, drop_rate=0., num_base=-1,
                 num_rel=None, wni=False, wsi=False, use_bn=True, ltr=True, comp_weights=None, agg_weights=None):
        super(SearchGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.act = act  # activation function
        self.device = None

        self.rel = nn.Parameter(torch.empty([num_rel, in_channels], dtype=torch.float))

        self.use_bn = use_bn
        self.ltr = ltr

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        # transform embedding of relations to next layer
        self.w_rel = self.get_param([in_channels, out_channels])
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel, num_base])
        else:
            self.rel_wt = None

        self.wni = wni
        self.wsi = wsi
        self.comp_weights = comp_weights
        self.agg_weights = agg_weights
        self.comp = CompMixOp()
        self.agg = AggMixOp()
        self.comb = CombMixOp(out_channels)
        self.act = ActMixOp()

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(
            edges.src['h'], self.rel[edge_type], self.comp_weights)  # [E, in_channel]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        # return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}
        return {'h': self.agg(nodes.mailbox['msg'], self.agg_weights)}

    def apply_node_func(self, nodes):
        return {'h': self.drop(nodes.data['h'])}

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm, comp_weights, agg_weights, comb_weights, act_weights):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """
        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm
        self.comp_weights = comp_weights
        self.agg_weights = agg_weights
        self.comb_weights = comb_weights
        self.act_weights = act_weights
        if self.rel_wt is None:
            self.rel.data = rel_repr
        else:
            # [num_rel*2, num_base] @ [num_base, in_c]
            self.rel.data = torch.mm(self.rel_wt, rel_repr)
        g.update_all(self.message_func, self.reduce_func, self.apply_node_func)

        if (not self.wni) and (not self.wsi):
            x = self.comb(g.ndata.pop('h'), torch.mm(self.comp(x, self.loop_rel, self.comp_weights), self.loop_w), self.comb_weights)*(1/3)
            # x = (g.ndata.pop('h') +
            #      torch.mm(self.comp(x, self.loop_rel, self.comp_weights), self.loop_w)) / 3
        # else:
        #     if self.wsi:
        #         x = g.ndata.pop('h') / 2
        #     if self.wni:
        #         x = torch.mm(self.comp(x, self.loop_rel), self.loop_w)

        if self.bias is not None:
            x = x + self.bias

        if self.use_bn:
            x = self.bn(x)

        if self.ltr:
            return self.act(x, self.act_weights), torch.matmul(self.rel.data, self.w_rel)
        else:
            return self.act(x, self.act_weights), self.rel.data