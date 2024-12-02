import torch
from torch.nn import Module, Linear

COMP_OPS = {
    'mult': lambda: MultOp(),
    'sub': lambda: SubOp(),
    'add': lambda: AddOp(),
    'ccorr': lambda: CcorrOp(),
    'rotate': lambda: RotateOp()
}

AGG_OPS = {
    'max': lambda: MaxOp(),
    'sum': lambda: SumOp(),
    'mean': lambda: MeanOp()
}

COMB_OPS = {
    'add': lambda out_channels: CombAddOp(out_channels),
    'mlp': lambda out_channels: CombMLPOp(out_channels),
    'concat': lambda out_channels: CombConcatOp(out_channels)
}

ACT_OPS = {
    'identity': lambda: torch.nn.Identity(),
    'relu': lambda: torch.nn.ReLU(),
    'tanh': lambda: torch.nn.Tanh(),
}

class CcorrOp(Module):
    def __init__(self):
        super(CcorrOp, self).__init__()

    def forward(self, src_emb, rel_emb):
        return self.comp(src_emb, rel_emb)

    def comp(self, h, edge_data):
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

        return ccorr(h, edge_data.expand_as(h))


class MultOp(Module):

    def __init__(self):
        super(MultOp, self).__init__()

    def forward(self, src_emb, rel_emb):
        # print('hr.shape', hr.shape)
        return src_emb * rel_emb


class SubOp(Module):
    def __init__(self):
        super(SubOp, self).__init__()

    def forward(self, src_emb, rel_emb):
        # print('hr.shape', hr.shape)
        return src_emb - rel_emb


class AddOp(Module):
    def __init__(self):
        super(AddOp, self).__init__()

    def forward(self, src_emb, rel_emb):
        # print('hr.shape', hr.shape)
        return src_emb + rel_emb


class RotateOp(Module):
    def __init__(self):
        super(RotateOp, self).__init__()

    def forward(self, src_emb, rel_emb):
        # print('hr.shape', hr.shape)
        return self.rotate(src_emb, rel_emb)

    def rotate(self, h, r):
        # re: first half, im: second half
        # assume embedding dim is the last dimension
        d = h.shape[-1]
        h_re, h_im = torch.split(h, d // 2, -1)
        r_re, r_im = torch.split(r, d // 2, -1)
        return torch.cat([h_re * r_re - h_im * r_im,
                          h_re * r_im + h_im * r_re], dim=-1)


class MaxOp(Module):
    def __init__(self):
        super(MaxOp, self).__init__()

    def forward(self, msg):
        return torch.max(msg, dim=1)[0]


class SumOp(Module):
    def __init__(self):
        super(SumOp, self).__init__()

    def forward(self, msg):
        return torch.sum(msg, dim=1)


class MeanOp(Module):
    def __init__(self):
        super(MeanOp, self).__init__()

    def forward(self, msg):
        return torch.mean(msg, dim=1)


class CombAddOp(Module):
    def __init__(self, out_channels):
        super(CombAddOp, self).__init__()

    def forward(self, self_emb, msg):
        return self_emb + msg


class CombMLPOp(Module):
    def __init__(self, out_channels):
        super(CombMLPOp, self).__init__()
        self.linear = Linear(out_channels, out_channels)

    def forward(self, self_emb, msg):
        return self.linear(self_emb + msg)


class CombConcatOp(Module):
    def __init__(self, out_channels):
        super(CombConcatOp, self).__init__()
        self.linear = Linear(2*out_channels, out_channels)

    def forward(self, self_emb, msg):
        return self.linear(torch.concat([self_emb,msg],dim=1))