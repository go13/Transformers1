from abc import ABC

from logging import getLogger
import torch
import torch.nn.functional as F

from .transformer_config import TransformerConfig
from .utils import DispatchingModule

from .abstract_transformer import TransformerEncoder, TransformerDecoder

logger = getLogger()


class SentimentalTransformer(DispatchingModule, ABC):

    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config
        self.n_layers = config.n_enc_layers

        self.te1 = TransformerEncoder(config, False)
        self.td1 = TransformerDecoder(config, True, False)

        self.nn1 = MultiLayerNN(config, config.input_seq_length * config.dim, config.hidden_dim, 1)

        self.output = self.nn1

    def learn(self, tensor, pred_mask, y):
        # scores = self.generate(tensor, pred_mask)

        mse_loss = torch.nn.MSELoss(reduction='sum')
        loss = mse_loss(tensor, y)

        return scores, loss

    def generate(self, tensor, pred_mask):
        # assert tensor.shape == (self.config.input_seq_length, self.config.batch_size, self.config.dim)
        # x = tensor[pred_mask.unsqueeze(-1).expand_as(tensor)].view(-1, self.config.dim)
        # scores = self.output.proj(x).view(-1, self.config.n_words)

        return scores

    def fwd(self, x1, len1, x2, len2):
        encoded1 = self.te1.fwd(x=x1, lengths=len1, causal=False)
        tensor = self.td1.fwd(x=x2, lengths=len2, causal=True, src_enc=encoded1.transpose(0, 1), src_len=len1)

        tensor = tensor.transpose(0, 1)

        tensor = self.nn1.fwd(x=tensor)

        return tensor


def build_sentimental_transformer(env, params):
    modules = {}

    config = TransformerConfig(params, env.id2word, False)

    modules['transformer'] = SentimentalTransformer(config)

    for k, v in modules.items():
        logger.info(f"Number of parameters ({k}): {sum([p.numel() for p in v.parameters() if p.requires_grad])}")

    assert not params.cpu

    if not params.cpu:
        for v in modules.values():
            v.cuda(config.my_device)

    # modules['transformer'].to_device(config.my_device)

    return modules
