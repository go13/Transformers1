from t3_karpathy.commons import BaseTransformerConfig
from t3_karpathy.token_codec import TokenCodec


class TransformerConfig(BaseTransformerConfig):

    def __init__(self, my_device='cuda', batch_size=64, block_size=32, n_embed=64, n_head=4, n_layer=4, learning_rate=1e-3):
        super().__init__(my_device, batch_size, block_size, n_embed, n_head, n_layer, learning_rate)
        self.token_codec = TokenCodec()
        self.vocab_size = self.token_codec.vocab_size

        # self.bottleneck_dim = params.bottleneck_dim
        # self.n_heads = params.n_heads
        # self.n_enc_layers = params.n_enc_layers
        # self.n_dec_layers = params.n_dec_layers
        # self.dropout = params.dropout
        #
        # assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'