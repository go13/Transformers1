import time

from t3_karpathy.commons import BaseTransformerConfig
from t3_karpathy.transformer_config import TransformerConfig


from collections import OrderedDict


from t3_karpathy.transformer_config import TransformerConfig
# todo create step embedding and leave only one trans layer and iterate it. while extract weights using attention in another transformer
# todo extract weights into separate transformer and learn layers to read write weights based on memory

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Optional, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import skip_init

from tqdm import tqdm


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(my_device, dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis.to(my_device)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.n_local_heads = config.n_head  # fs_init.get_model_parallel_world_size()
        self.head_dim = config.n_embd // config.n_head

        self.wq = skip_init(
            nn.Linear,
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wk = skip_init(
            nn.Linear,
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wv = skip_init(
            nn.Linear,
            config.n_embd,
            config.n_head * self.head_dim,
            bias=False,
        )
        self.wo = skip_init(
            nn.Linear,
            config.n_head * self.head_dim,
            config.n_embd,
            bias=False,
        )

        self.cache_k = torch.zeros(
            (config.batch_size, config.max_seq_len, self.n_local_heads, self.head_dim), device=config.my_device
        )
        self.cache_v = torch.zeros(
            (config.batch_size, config.max_seq_len, self.n_local_heads, self.head_dim), device=config.my_device
        )

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # self.cache_k = self.cache_k.to(xq)
        # self.cache_v = self.cache_v.to(xq)
        #
        # self.cache_k[:bsz, start_pos: start_pos + seqlen] = xk
        # self.cache_v[:bsz, start_pos: start_pos + seqlen] = xv
        #
        # keys = self.cache_k[:bsz, : start_pos + seqlen]
        # values = self.cache_v[:bsz, : start_pos + seqlen]

        keys = xk
        values = xv

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, slen, cache_len + slen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(
            1, 2
        ).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            multiple_of: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = skip_init(
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = skip_init(
            nn.Linear,
            hidden_dim,
            dim,
            bias=False,
        )
        self.w3 = skip_init(
            nn.Linear,
            dim,
            hidden_dim,
            bias=False,
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_head
        self.dim = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=config.n_embd, hidden_dim=4 * config.n_embd, multiple_of=config.multiple_of
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.n_embd, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class LlamaBlockSequence(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(layer_id, config) for layer_id in range(config.n_layer)])

    def forward(self, h, start_pos, freqs_cis, mask):
        for block in self.blocks:
            h = block(h, start_pos, freqs_cis, mask)
        return h


class LlamaTransformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layer

        self.tok_embeddings = skip_init(
            nn.Embedding,
            config.vocab_size,
            config.n_embd
        )
        self.layers = LlamaBlockSequence(config)

        self.norm = RMSNorm(config.n_embd, eps=config.norm_eps)
        self.output = skip_init(
            nn.Linear,
            config.n_embd,
            config.vocab_size,
            bias=False
        )

        self.freqs_cis = precompute_freqs_cis(
            config.my_device, self.config.n_embd // self.config.n_head, self.config.max_seq_len * 2
        )

    def forward_vs_target(self, idx, targets):
        logits = self.forward(idx)

        b, t, c = logits.shape
        logits_view = logits.view(b * t, c)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits_view, targets)

        return logits_view, loss

    # @torch.inference_mode()
    def forward(self, tokens):
        # def forward(self, tokens: torch.Tensor, start_pos: int):
        start_pos = 0

        _bsz, seqlen = tokens.shape

        h = self.tok_embeddings(tokens)

        # self.freqs_cis = self.freqs_cis
        freqs_cis = self.freqs_cis[start_pos: start_pos + seqlen]

        # h = h.cuda()

        # mask = None
        # if seqlen > 1:
        mask = torch.full(
            (1, 1, seqlen, seqlen), float("-inf"), device=self.config.my_device
        )
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        # if mask is not None:
        # mask = mask.cuda()

        h = self.layers(h, start_pos, freqs_cis, mask)

        h = self.norm(h)

        del mask
        torch.cuda.empty_cache()

        # output = self.output(h[:, -1, :])  # only compute last logits
        output = self.output(h)  # only compute last logits
        return output.float()

class AbstractRunner(object):
    def __init__(self, config: TransformerConfig, model):
        self.model = model.to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x):
        return self.model(x)

    def learn(self, x, y):
        self.model.train()
        out, loss = self.model.forward_vs_target(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # Add retain_graph=True here
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        t = time.time()
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                t_taken = time.time() - t
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}, time/iter {t_taken / self.config.eval_interval}")
                t = time.time()

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))

    def generate(self, *args):
        raise NotImplementedError()


class LlamaRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, LlamaTransformer(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)
