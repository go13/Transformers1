from abc import ABC

import torch.nn as nn
import torch.nn.functional as F

from .utils import DispatchingModule


class MultiLayerNN(DispatchingModule, ABC):

    def __init__(self, config, in_dim, dim_hidden, out_dim, hidden_layers=2):
        super().__init__()
        self.bs = config.batch_size
        self.dropout = config.dropout
        self.hidden_layers = hidden_layers

        self.lin_in = nn.Linear(in_dim, dim_hidden)
        self.norm_in = nn.LayerNorm(dim_hidden, eps=1e-12)

        self.lin = nn.ModuleList()
        self.norm = nn.ModuleList()
        for i in range(hidden_layers):
            self.lin.append(nn.Linear(dim_hidden, dim_hidden))
            self.norm.append(nn.LayerNorm(dim_hidden, eps=1e-12))

        self.lin_out = nn.Linear(dim_hidden, out_dim)

    def fwd(self, x):
        x = x.reshape(self.bs, -1)

        x = self.lin_in(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.norm_in(x)

        for i in range(self.hidden_layers):
            x = self.lin[i](x)
            x = F.relu(x)
            x = self.norm[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin_out(x)

        return x