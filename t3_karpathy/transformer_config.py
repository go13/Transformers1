import torch
from t3_karpathy.commons.commons import BaseTransformerConfig
from t3_karpathy.token_codec import TokenCodec


class TransformerConfig(BaseTransformerConfig):

    def __init__(self, my_device='cuda', precision=torch.bfloat16, batch_size=128, block_size=256, n_embed=16, n_head=2, n_layer=4, learning_rate=1e-3):
        super().__init__(my_device, precision, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.token_codec = TokenCodec()
        self.vocab_size = self.token_codec.vocab_size