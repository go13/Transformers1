from abc import ABC
import torch

from logging import getLogger
import torch.nn.functional as F

from .transformer_config import TransformerConfig
from .utils import DispatchingModule
from src.utils import to_cuda

from .abstract_transformer import TransformerEncoder

logger = getLogger()


class EncoderTransformer(DispatchingModule, ABC):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.n_layers = config.n_enc_layers

        self.te1 = TransformerEncoder(config, False)
        # self.td1 = TransformerDecoder(config, True, False)

        self.output = self.te1

    def learn(self, x1, y, len1):
        # self.train()

        x1, y, len1 = to_cuda(self.config.my_device, x1, y, len1)

        # len1 = torch.full((x1.shape[1],), x1.shape[0], device=self.config.my_device)

        pred_mask = self.get_pred_mask(len1)
        y = y[1:].masked_select(pred_mask[:-1])

        tensor = self.fwd(x1=x1, len1=len1)

        selected = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)]

        x = selected.view(-1, self.config.dim)

        scores = self.output.proj(x).view(-1, self.config.n_words)

        loss = F.cross_entropy(scores, y, reduction='mean')

        return scores, loss

    def generate(self, tensor, pred_mask):
        # assert tensor.shape == (self.config.input_seq_length, self.config.batch_size, self.config.dim)
        x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.config.dim)
        scores = self.output.proj(x).view(-1, self.config.n_words)

        return scores

    def fwd(self, x1, len1):
        tensor = self.te1.fwd(x=x1, lengths=len1, causal=False)
        # tensor = self.td1.fwd(x=x2, lengths=len2, causal=True, src_enc=encoded1.transpose(0, 1), src_len=len1)

        return tensor

    def get_pred_mask(self, len1):
        alen = torch.arange(len1.max(), dtype=torch.long, device=self.config.my_device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        return pred_mask


def build_transformer(env, params):
    modules = {}

    config = TransformerConfig(params, env.id2word, False)

    modules['transformer'] = EncoderTransformer(config)

    for k, v in modules.items():
        logger.info(f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}")

    assert not params.cpu

    if not params.cpu:
        for v in modules.values():
            v.cuda(config.my_device)

    return modules
