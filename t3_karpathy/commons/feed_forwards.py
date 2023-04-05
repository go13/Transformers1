from torch import nn as nn

from t3_karpathy.commons.commons import BaseTransformerConfig


class SentimentalFeedForward(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        inp_size = config.n_embed * config.block_size
        hidden_size = inp_size // 2  # config.hidden_size # * config.block_size
        dropout = config.dropout
        out_size = 1

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size, out_size),
            # FeedForward(inp_size, hidden_size, out_size, dropout),
        )

    def forward(self, x):
        return self.net(x)


class LinearFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, out_n_embd, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class GeluFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ReluFeedForward(nn.Module):
    def __init__(self, inp_n_embd, hidden_n_embd, out_n_embd, dropout, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_n_embd, hidden_n_embd, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_n_embd, out_n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
