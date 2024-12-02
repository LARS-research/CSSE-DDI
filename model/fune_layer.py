import torch
from torch import nn
import dgl
import dgl.function as fn
from model.genotypes import COMP_PRIMITIVES, AGG_PRIMITIVES, COMB_PRIMITIVES, ACT_PRIMITIVES
from model.operations import *


class CompOp(nn.Module):
    def __init__(self, primitive):
        super(CompOp, self).__init__()
        self._op = COMP_OPS[primitive]()

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, src_emb, rel_emb):
        return self._op(src_emb, rel_emb)


class AggOp(nn.Module):
    def __init__(self, primitive):
        super(AggOp, self).__init__()
        self._op = AGG_OPS[primitive]()

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, msg):
        return self._op(msg)


class CombOp(nn.Module):
    def __init__(self, primitive, out_channels):
        super(CombOp, self).__init__()
        self._op = COMB_OPS[primitive](out_channels)

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, self_emb, msg):
        return self._op(self_emb, msg)


class ActOp(nn.Module):
    def __init__(self, primitive):
        super(ActOp, self).__init__()
        self._op = ACT_OPS[primitive]()

    def reset_parameters(self):
        self._op.reset_parameters()

    def forward(self, emb):
        return self._op(emb)

def act_map(act):
    if act == "identity":
        return lambda x: x
    elif act == "elu":
        return torch.nn.functional.elu
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "tanh":
        return torch.tanh
    elif act == "relu":
        return torch.nn.functional.relu
    elif act == "relu6":
        return torch.nn.functional.relu6
    elif act == "softplus":
        return torch.nn.functional.softplus
    elif act == "leaky_relu":
        return torch.nn.functional.leaky_relu
    else:
        raise Exception("wrong activate function")


class SearchedGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, drop_rate=0., num_base=-1,
                 num_rel=None, wni=False, wsi=False, use_bn=True, ltr=True, comp=None, agg=None, comb=None, act=None):
        super(SearchedGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.comp_op = comp
        self.act = act_map(act)  # activation function
        self.agg_op = agg
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
        # self.comp = CompOp(comp)
        # self.agg = AggOp(agg)
        self.comb = CombOp(comb, out_channels)
        # self.act = ActOp(act)

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(
            edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        # return {'h': torch.sum(nodes.mailbox['msg'], dim=1)}
        return {'h': self.agg(nodes.mailbox['msg'])}

    def apply_node_func(self, nodes):
        return {'h': self.drop(nodes.data['h'])}

    def comp(self, h, edge_data):
        # def com_mult(a, b):
        #     r1, i1 = a[..., 0], a[..., 1]
        #     r2, i2 = b[..., 0], b[..., 1]
        #     return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)
        #
        # def conj(a):
        #     a[..., 1] = -a[..., 1]
        #     return a
        #
        # def ccorr(a, b):
        #     # return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))
        #     return torch.fft.irfftn(torch.conj(torch.fft.rfftn(a, (-1))) * torch.fft.rfftn(b, (-1)), (-1))

        def com_mult(a, b):
            r1, i1 = a.real, a.imag
            r2, i2 = b.real, b.imag
            real = r1 * r2 - i1 * i2
            imag = r1 * i2 + i1 * r2
            return torch.complex(real, imag)

        def conj(a):
            a.imag = -a.imag
            return a

        def ccorr(a, b):
            return torch.fft.irfft(com_mult(conj(torch.fft.rfft(a)), torch.fft.rfft(b)), a.shape[-1])

        def rotate(h, r):
            # re: first half, im: second half
            # assume embedding dim is the last dimension
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im,
                              h_re * r_im + h_im * r_re], dim=-1)

        if self.comp_op == 'mult':
            return h * edge_data
        elif self.comp_op == 'add':
            return h + edge_data
        elif self.comp_op == 'sub':
            return h - edge_data
        elif self.comp_op == 'ccorr':
            return ccorr(h, edge_data.expand_as(h))
        elif self.comp_op == 'rotate':
            return rotate(h, edge_data)
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
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
        if self.rel_wt is None:
            self.rel.data = rel_repr
        else:
            # [num_rel*2, num_base] @ [num_base, in_c]
            self.rel.data = torch.mm(self.rel_wt, rel_repr)
        if self.agg_op == 'max':
            g.update_all(self.message_func, fn.max(msg='msg', out='h'), self.apply_node_func)
        elif self.agg_op == 'mean':
            g.update_all(self.message_func, fn.mean(msg='msg', out='h'), self.apply_node_func)
        elif self.agg_op == 'sum':
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.apply_node_func)
        # g.update_all(self.message_func, self.reduce_func, self.apply_node_func)

        if (not self.wni) and (not self.wsi):
            x = self.comb(g.ndata.pop('h'), torch.mm(self.comp(x, self.loop_rel), self.loop_w))*(1/3)
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
            return self.act(x), torch.matmul(self.rel.data, self.w_rel)
        else:
            return self.act(x), self.rel.data