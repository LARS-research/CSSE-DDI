import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SortPooling, SumPooling
from dgl.nn.pytorch import GraphConv, SAGEConv
from dgl import NID, EID


class SEAL_GCN(nn.Module):
    def __init__(self, num_ent, num_rel, init_dim, gcn_dim, embed_dim, n_layer, loss_type, max_z=1000):
        super(SEAL_GCN, self).__init__()

        if loss_type == 'ce':
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == 'bce':
            self.loss = nn.BCELoss(reduce=False)
        elif loss_type == 'bce_logits':
            self.loss = nn.BCEWithLogitsLoss()
        self.init_embed = self.get_param([num_ent, init_dim])
        self.init_rel = self.get_param([num_rel, init_dim])
        self.z_embedding = nn.Embedding(max_z, init_dim)
        init_dim += init_dim

        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(init_dim, gcn_dim, allow_zero_in_degree=True))
        for _ in range(n_layer - 1):
            self.layers.append(GraphConv(gcn_dim, gcn_dim, allow_zero_in_degree=True))

        self.linear_1 = nn.Linear(embed_dim, embed_dim)
        self.linear_2 = nn.Linear(embed_dim, num_rel)
        self.pooling = SumPooling()

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

    def forward(self, g, z):
        x, r = self.init_embed[g.ndata[NID]], self.init_rel
        z_emb = self.z_embedding(z)
        x = torch.cat([x, z_emb], 1)
        for layer in self.layers[:-1]:
            x = layer(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](g, x)

        x = self.pooling(g, x)
        x = F.relu(self.linear_1(x))
        F.dropout(x, p=0.5, training=self.training)
        x = self.linear_2(x)

        return x