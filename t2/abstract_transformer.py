from abc import ABC

from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import Embedding, create_sinusoidal_embeddings
from .multi_head_attention import MultiHeadAttention
from .utils import N_MAX_POSITIONS, DispatchingModule
from .transformer_config import TransformerConfig
from .transformer_ffn import TransformerFFN

logger = getLogger()


class AbstractTransformer(DispatchingModule, ABC):

    def __init__(self, config, is_decoder):
        super().__init__()

        self.config = config
        self.is_decoder = is_decoder


class TransformerEncoder(AbstractTransformer, ABC):

    def __init__(self, config: TransformerConfig, is_raw_input: bool):
        super().__init__(config, False)
        self.n_layers = config.n_enc_layers
        self.is_raw_input = is_raw_input
        # embeddings
        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.config.dim)
        if config.sinusoidal_embeddings:
            create_sinusoidal_embeddings(N_MAX_POSITIONS, self.config.dim, out=self.position_embeddings.weight)
        if not is_raw_input:
            self.embeddings = Embedding(self.config.n_words, self.config.dim, padding_idx=self.config.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.config.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.config.n_heads, self.config.dim, dropout=self.config.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.config.dim, eps=1e-12))
            self.ffns.append(TransformerFFN(self.config.dim, self.config.hidden_dim, self.config.dim, dropout=self.config.dropout))
            self.layer_norm2.append(nn.LayerNorm(self.config.dim, eps=1e-12))

    def fwd(self, x, lengths, causal, positions=None, cache=None, previous_state=None):
        # check inputs
        # bs, slen = x.size()
        slen, bs = x.size()
        x = x.transpose(0, 1)  # batch size as dimension 0
        if self.is_raw_input:
            slen = self.config.input_seq_length
            x = x.reshape(self.config.batch_size, slen, self.config.dim)
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen

        assert previous_state is None or cache is None

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # embeddings
        if previous_state is None:
            if not self.is_raw_input:
                tensor = self.embeddings(x)
            else:
                tensor = x
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
            tensor = self.layer_norm_emb(tensor)
            tensor = F.dropout(tensor, p=self.config.dropout, training=self.training)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        else:
            assert previous_state.shape == (slen, bs, self.config.dim)
            tensor = previous_state.transpose(0, 1)

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.config.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # if self.config.nn_output and self.with_output:
        #     tensor = self.output[0](tensor.reshape(bs, -1))

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor


class TransformerDecoder(AbstractTransformer, ABC):

    def __init__(self, config, is_last, is_raw_input):
        super().__init__(config, True)
        self.is_raw_input = is_raw_input
        self.n_layers = config.n_dec_layers

        self.position_embeddings = Embedding(N_MAX_POSITIONS, self.config.dim)
        self.embeddings = Embedding(self.config.n_words, self.config.dim, padding_idx=self.config.pad_index)
        self.layer_norm_emb = nn.LayerNorm(self.config.dim, eps=1e-12)

        # transformer layers
        self.attentions = nn.ModuleList()
        self.layer_norm1 = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.layer_norm2 = nn.ModuleList()

        self.layer_norm15 = nn.ModuleList()
        self.encoder_attn = nn.ModuleList()

        for layer_id in range(self.n_layers):
            self.attentions.append(MultiHeadAttention(self.config.n_heads, self.config.dim, dropout=self.config.attention_dropout))
            self.layer_norm1.append(nn.LayerNorm(self.config.dim, eps=1e-12))
            self.layer_norm15.append(nn.LayerNorm(self.config.dim, eps=1e-12))
            self.encoder_attn.append(MultiHeadAttention(self.config.n_heads, self.config.dim, dropout=self.config.attention_dropout))
            self.ffns.append(TransformerFFN(self.config.dim, self.config.hidden_dim, self.config.dim, dropout=self.config.dropout))
            self.layer_norm2.append(nn.LayerNorm(self.config.dim, eps=1e-12))

        self.proj = nn.Linear(self.config.dim, config.n_words, bias=True)
        if config.share_inout_emb and is_last:
            self.proj.weight = self.embeddings.weight

    def fwd(self, x, lengths, causal, src_enc, src_len, positions=None, cache=None, previous_state=None):

        slen, bs = x.size()
        x = x.transpose(0, 1)  # batch size as dimension 0
        if self.is_raw_input:
            slen = self.config.input_seq_length
            x = x.reshape(self.config.batch_size, slen, self.config.dim)
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        assert (src_enc is None) == (src_len is None)
        assert self.is_decoder and src_enc is not None
        assert src_enc.size(0) == bs
        assert previous_state is None or cache is None

        # generate masks
        mask, attn_mask = get_masks(slen, lengths, causal)
        src_mask = torch.arange(src_len.max(), dtype=torch.long, device=self.config.my_device) < src_len[:, None]

        # positions
        if positions is None:
            positions = x.new(slen).long()
            positions = torch.arange(slen, out=positions).unsqueeze(0)
        else:
            assert positions.size() == (slen, bs)
            positions = positions.transpose(0, 1)

        # do not recompute cached elements
        if cache is not None:
            _slen = slen - cache['slen']
            x = x[:, -_slen:]
            positions = positions[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]

        # all layer outputs
        if TransformerConfig.STORE_OUTPUTS and not self.training:
            self.outputs = []

        # embeddings
        if previous_state is None:
            if not self.is_raw_input:
                tensor = self.embeddings(x)
            else:
                tensor = x
            tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
            tensor = self.layer_norm_emb(tensor)
            tensor = F.dropout(tensor, p=self.config.dropout, training=self.training)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        else:
            assert previous_state.shape == (slen, bs, self.dim)
            tensor = previous_state.transpose(0, 1)

        if TransformerConfig.STORE_OUTPUTS and not self.training:
            self.outputs.append(tensor.detach().cpu())

        # transformer layers
        for i in range(self.n_layers):
            # self attention
            attn = self.attentions[i](tensor, attn_mask, cache=cache)
            attn = F.dropout(attn, p=self.config.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm1[i](tensor)

            # encoder attention (for decoder only)
            attn = self.encoder_attn[i](tensor, src_mask, kv=src_enc, cache=cache)
            attn = F.dropout(attn, p=self.config.dropout, training=self.training)
            tensor = tensor + attn
            tensor = self.layer_norm15[i](tensor)

            # FFN
            tensor = tensor + self.ffns[i](tensor)
            tensor = self.layer_norm2[i](tensor)

            tensor *= mask.unsqueeze(-1).to(tensor.dtype)

        # update cache length
        if cache is not None:
            cache['slen'] += tensor.size(1)

        # move back sequence length to dimension 0
        tensor = tensor.transpose(0, 1)

        return tensor

def get_masks(slen, lengths, causal):
    """
    Generate hidden states mask, and optionally an attention mask.
    """
    assert lengths.max().item() <= slen
    bs = lengths.size(0)
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)
    mask = alen < lengths[:, None]

    # attention mask is the same as mask, or triangular inferior attention (causal)
    if causal:
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None]
    else:
        attn_mask = mask

    # sanity check
    assert mask.size() == (bs, slen)
    assert causal is False or attn_mask.size() == (bs, slen, slen)

    return mask, attn_mask
