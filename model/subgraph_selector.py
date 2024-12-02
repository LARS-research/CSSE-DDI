import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SubgraphSelector(nn.Module):
    def __init__(self, args):
        super(SubgraphSelector,self).__init__()
        self.args = args
        self.temperature = self.args.temperature
        self.num_layers = self.args.ss_num_layer
        self.cat_type = self.args.combine_type
        if self.cat_type == 'mult':
            in_channels = self.args.ss_input_dim
        else:
            in_channels = self.args.ss_input_dim * 2
        hidden_channels = self.args.ss_hidden_dim
        self.trans = nn.ModuleList()
        for i in range(self.num_layers - 1):
            if i == 0:
                self.trans.append(nn.Linear(in_channels, hidden_channels, bias=False))
            else:
                self.trans.append(nn.Linear(hidden_channels, hidden_channels, bias=False))
        self.trans.append(nn.Linear(hidden_channels, 1, bias=False))

    def forward(self, x, mode='argmax', search_algorithm='darts'):
        for layer in self.trans[:-1]:
            x = layer(x)
            x= F.relu(x)
        x = self.trans[-1](x)
        x = torch.squeeze(x,dim=2)
        if search_algorithm == 'darts':
            arch_set = torch.softmax(x/self.temperature,dim=1)
        elif search_algorithm == 'snas':
            arch_set = self._get_categ_mask(x)
        if mode == 'argmax':
            device = arch_set.device
            n, c = arch_set.shape
            eyes_atten = torch.eye(c).to(device)
            atten_ , atten_indice = torch.max(arch_set, dim=1)
            arch_set = eyes_atten[atten_indice]
        return arch_set
        # raise NotImplementedError

    def _get_categ_mask(self, alpha):
        # log_alpha = torch.log(alpha)
        log_alpha = alpha
        u = torch.zeros_like(log_alpha).uniform_()
        softmax = torch.nn.Softmax(-1)
        one_hot = softmax((log_alpha + (-((-(u.log())).log()))) / self.temperature)
        return one_hot