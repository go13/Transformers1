import torch
import torch.nn as nn
from flash_attn.modules.mha import MHA
from torch.nn import functional as F

from t3_karpathy.commons.commons import AbstractRunner, BaseTransformerConfig
from t3_karpathy.commons.feed_forwards import GeluFeedForward
from t4.generic_dataloader import AbstractDataLoader


def distance_triangle(n, my_device):
    arange_matrix = torch.arange(n, device=my_device).view(-1, 1) - torch.arange(n, device=my_device).view(1, -1)
    lower_triangular = torch.tril(arange_matrix)
    return lower_triangular


class PositionalEmbedding(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(config.n_embed, config.n_embed, config.n_embed, config.dropout)
        self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed)

    def forward(self, b, t):
        pos_embedding_arrange = torch.arange(t, device=self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange).repeat(b, 1, 1)  # (B,T,C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        # pos_emb = self.dropout(pos_emb)

        # pos_emb = pos_emb.unsqueeze(1).repeat(1, t, 1, 1)  # (B,T,C) -> (B,T,T,C)
        # k = pos_emb
        # q = pos_emb.transpose(1, 2)
        # pos_emb = torch.cat([k, q], dim=-1)  # (B,T,T,C)

        # return k + q
        return pos_emb


class DistancePositionalEmbedding(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.position_embedding_ff = GeluFeedForward(
            config.n_embed,
            config.n_embed,
            config.n_embed,
            config.dropout
        )
        # self.position_embedding_ff_ln = nn.LayerNorm(config.n_embed * 2)

    def forward(self, b):
        pos_embedding_arrange = distance_triangle(self.config.block_size, self.config.my_device)
        pos_emb = self.position_embedding_table(pos_embedding_arrange)
        pos_emb = pos_emb.repeat(b, 1, 1, 1)  # (B, T, T, C)
        pos_emb = self.position_embedding_ff.forward(pos_emb)
        # pos_emb = self.position_embedding_ff_ln(pos_emb)
        return pos_emb


class FlashMultiHeadAttention(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()
        assert torch.bfloat16 == config.precision, 'only bfloat16 is supported - checked 20 aug 23'

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
            causal=causal  # auto-regressive or not
        )

    def forward(self, x, pos_emb, pos_dist_emb):
        x = x + pos_emb
        out = self.flash_mha(x)[0]
        return out


class Block(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()

        self.sa = FlashMultiHeadAttention(config, causal=causal)

        self.ln2 = nn.LayerNorm(config.n_embed)

        self.ffwd = GeluFeedForward(config.n_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

    def forward(self, x, pos_emb, pos_dist_emb):
        x = x + self.sa.forward(x, pos_emb, pos_dist_emb)
        x = x + self.ffwd.forward(self.ln2(x))
        return x


class BlockSequence(nn.Module):
    def __init__(self, config: BaseTransformerConfig, causal=True):
        super().__init__()
        self.blocks = nn.Sequential(*[Block(config, causal) for _ in range(config.n_layer)])

    def forward(self, x, pos_emb, pos_dist_emb):
        for block in self.blocks:
            x = block(x, pos_emb, pos_dist_emb)
        return x


class TransformerConfig(BaseTransformerConfig):
    def __init__(self, input_embed, output_num, vocab_size, my_device='cuda', precision=torch.bfloat16, batch_size=128,
                 block_size=256,
                 n_embed=16, n_head=2, n_layer=4, learning_rate=1e-3, causal=True):
        super().__init__(my_device, precision, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.input_embed = input_embed
        self.output_num = output_num
        self.causal = causal
        self.vocab_size = vocab_size


class TransformerModel(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.out_stock_embedding = nn.Embedding(config.output_num, config.n_embed)

        self.out_stock_ffwd = GeluFeedForward(config.n_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

        self.pos_emb1 = PositionalEmbedding(config)
        self.pos_dist_emb1 = DistancePositionalEmbedding(config)

        self.ffwd1 = GeluFeedForward(config.input_embed, config.hidden_size, config.n_embed, config.dropout, bias=False)

        self.t1 = BlockSequence(config, causal=config.causal)

        self.ffwd2 = GeluFeedForward(config.n_embed, config.hidden_size, config.vocab_size,
                                     config.dropout, bias=False)

    def forward_vs_target(self, inp, targets):
        output = self.forward(inp)

        b, t, c = output.shape
        # print(output.shape)
        logits_view = output.view(b * t * c // self.config.vocab_size, self.config.vocab_size)
        targets = targets.view(-1)
        # targets = targets.view(b * t, -1)
        # print(logits_view.shape)
        # print(targets.shape)

        loss = F.cross_entropy(logits_view, targets)

        return output, loss

    def forward(self, inp):
        x = inp[0]
        stock_ix = inp[1]

        out_stock_emb = self.out_stock_embedding(stock_ix)
        out_stock_emb = self.out_stock_ffwd.forward(out_stock_emb)
        # x = self.tok_emb.forward(x)

        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])

        b, t, c = x.shape

        x = self.ffwd1.forward(x)

        pos_emb = self.pos_emb1.forward(b, t) + out_stock_emb

        # print(pos_emb.shape, out_stock_emb.shape)

        # pos_dist_emb = self.pos_dist_emb1.forward(b)
        # print(pos_dist_emb.shape)
        x = self.t1.forward(x, pos_emb, None)

        x = self.ffwd2.forward(x)

        return x

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # get the predictions
            logits = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class InOutDataloader(AbstractDataLoader):
    def __init__(self, config: BaseTransformerConfig, in_data, out_data):
        super().__init__(config)
        self.in_data = in_data.to(self.config.my_device).to(config.precision)
        self.out_data = out_data.to(self.config.my_device)#.to(config.precision)
        self.config = config

        n = int(0.9 * len(self.in_data))  # first 90% will be trained, rest val
        self.in_train_data = self.in_data[:n]
        self.in_val_data = self.in_data[n:]

        self.out_train_data = self.out_data[:n]
        self.out_val_data = self.out_data[n:]

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        in_data = self.in_train_data if split == 'train' else self.in_val_data
        out_data = self.out_train_data if split == 'train' else self.out_val_data

        # print(out_data.shape)
        batch_ix = torch.randint(len(in_data) - self.config.block_size, (self.config.batch_size,))

        stock_idx = []
        x = []
        y = []
        for i in batch_ix:
            stock_out_ix = torch.randint(out_data.shape[-1], (1,))

            x.append(in_data[i:i + self.config.block_size])
            y.append(out_data[i + 1:i + self.config.block_size + 1, stock_out_ix])
            stock_idx.append(stock_out_ix)

        x = torch.stack(x)
        y = torch.stack(y)
        stock_idx = torch.stack(stock_idx)

        # x = torch.stack([in_data[i:i + self.config.block_size] for i in ix])
        # y = torch.stack([out_data[i + 1:i + self.config.block_size + 1] for i in ix])

        x, y, stock_idx = x.to(self.config.my_device), y.to(self.config.my_device), stock_idx.to(self.config.my_device)

        # print(x.shape, y.shape)
        return (x, stock_idx), y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')


class TransformerRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig, in_data, out_data):
        super().__init__(
            config,
            TransformerModel(config),
            InOutDataloader(config, in_data, out_data)
        )
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
