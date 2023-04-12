import torch
import torch.nn as nn
from torch.nn import functional as F
from flash_attn.modules.mha import MHA

from t3_karpathy.commons.commons import AbstractRunner, BaseTransformerConfig, AbstractDataLoader, \
    AbstractAccumulativeTrainer, AbstractCodec, TimeseriesFeedForward
from t3_karpathy.commons.feed_forwards import GeluFeedForward
from t3_karpathy.transformer_config import TransformerConfig


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        assert torch.bfloat16 == config.precision, 'only bfloat16 is supported'
        self.config = config
        # MHA + rotary requires flash-attention\csrc\rotary>pip install .
        self.flash_mha = MHA(
            embed_dim=config.n_embed,  # total channels (= num_heads * head_dim)
            num_heads=config.n_head,
            device=config.my_device,
            dtype=config.precision,
            dropout=config.dropout,
            use_flash_attn=True,
            return_residual=True,
            dwconv=True,
            rotary_emb_dim=config.head_size,
            causal=False  # auto-regressive or not
        )

    def forward(self, x):
        inp = x
        out = self.flash_mha(inp)[0]
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        self.sa = FlashMultiHeadAttention(config)

        dropout = config.dropout
        hidden_emb = config.hidden_size
        n_embed = config.n_embed
        out_emb = config.n_embed

        self.ln2 = nn.LayerNorm(n_embed)
        self.ffwd = GeluFeedForward(n_embed, hidden_emb, out_emb, dropout, bias=False)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(self.ln2(x))
        return x

    @staticmethod
    def create(config: BaseTransformerConfig):
        return Block(config)


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[Block.create(config) for _ in range(config.n_layer)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TransformerSentimentalModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # self.blocks = BlockSequence(config) uncomment
        self.ln_f = nn.LayerNorm(config.n_embed)  # final layer norm

        dropout = config.dropout
        inp_size = config.n_embed * config.block_size
        hidden_size = config.hidden_size
        out_size = 1

        # self.out = nn.Linear(inp_size, out_size, bias=False)
        # self.out = TimeseriesFeedForward(inp_size, hidden_size, out_size, dropout, bias=True)
        # self.out = TimeseriesFeedForward(inp_size, inp_size * 2, hidden_size, dropout, bias=True)
        # self.out2 = TimeseriesFeedForward(hidden_size, hidden_size, out_size, dropout, bias=True)

        self.out = nn.Sequential(
            TimeseriesFeedForward(inp_size, inp_size, hidden_size, dropout, bias=True),
            nn.Dropout(dropout),
            TimeseriesFeedForward(hidden_size, hidden_size, out_size, dropout, bias=True)
        )

    def forward_vs_target(self, x, targets):
        logits = self.forward(x)

        # b, c = logits.shape
        # logits_view = logits.view(b * t, c)
        # targets = targets.view(b * t)
        # output = F.cross_entropy(logits, targets)

        mse_loss = torch.nn.MSELoss(reduction='mean')
        loss = mse_loss(logits, targets)

        return logits, loss

    def forward(self, x):
        b, t, c = x.shape
        # x = self.blocks(x)  # (B,T,C)

        # x = self.ln_f(x)  # (B,T,C)
        x = x.reshape(b, -1)
        logits = self.out(x)  # (B,T,1)
        logits = logits.reshape(b)

        return logits


class TransformerSentimentalRunner(AbstractRunner):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config, TransformerSentimentalModel(config), None)
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)


class TransformerCodec(AbstractCodec):
    def __init__(self, seq_len, num_channels, config):
        super().__init__()
        self.config = config
        self.seq_len = seq_len
        self.num_channels = num_channels

    def encode(self, t: AbstractRunner):
        weights = t.get_weights_as_tensor()
        return self.encode_weights(weights)

    def encode_weights(self, weights_tensor):
        weights_tensor = weights_tensor.to(self.config.my_device, dtype=self.config.precision)
        target_size = self.seq_len * self.num_channels

        padding_size = target_size - weights_tensor.numel()

        padding = torch.zeros(padding_size, device=weights_tensor.device, dtype=weights_tensor.dtype)

        padded_input = torch.cat([padding, weights_tensor], dim=-1)

        w = padded_input.view(-1, 8)#.unsqueeze(0)

        return w

    def decode_into(self, weights_tensor, t: AbstractRunner) -> AbstractRunner:
        return t.set_weights_as_tensor(weights_tensor) # will it be even used?


class TransformerSentimentalAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: TransformerSentimentalRunner = TransformerSentimentalRunner(config)
        self.codec = TransformerCodec(config.block_size, config.n_embed, config)
        self.data_x = []
        self.data_y = []
        # self.data_dict = dict()

        self.last_data_x = []
        self.last_data_y = []

    def get_batch(self, data_x, data_y):
        ix = torch.randint(len(data_x), (self.config.batch_size,))
        x = torch.stack([torch.tensor(data_x[i]) for i in ix])
        y = torch.stack([torch.tensor(data_y[i]) for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device, dtype=self.config.precision)
        return x, y

    def add_sample(self, x, y):
        x = self.codec.encode(x)
        y = y.to(self.config.my_device, dtype=self.config.precision)

        self.last_data_x += [x]
        self.last_data_y += [y]

        # if x in self.data_dict:
        #     self.data_dict[x] += 1
        #     return self.data_dict[x]

        self.data_x += [x]
        self.data_y += [y]

        # self.data_dict[x] = 1
        #
        # return 1

    def predict_list(self, lst: list):
        lst = [torch.tensor(self.codec.encode_weights(x)).to(self.config.my_device, dtype=self.config.precision) for x in lst]
        x = torch.stack(lst).to(self.config.my_device, dtype=self.config.precision)
        out = self.runner.forward(x)
        return out.tolist()

    def predict(self, x):
        encoded_x = self.codec.encode_weights(x)
        x = torch.tensor(encoded_x).to(self.config.my_device)
        x = x.reshape(1, x.shape[0])
        out = self.runner.forward(x)
        return out.item()

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch(self.data_x, self.data_y)
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)

    def train_recent(self, n=1):
        for i in range(n):
            x, y = self.get_batch(self.last_data_x, self.last_data_y)
            o, loss = self.runner.learn(x, y)

        self.last_data_x = []
        self.last_data_y = []