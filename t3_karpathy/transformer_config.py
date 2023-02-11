from t3_karpathy.token_codec import TokenCodec


class TransformerConfig:

    def __init__(self, my_device='cuda', batch_size=16, block_size=32, n_embed=64, n_head=4, n_layer=4, learning_rate=1e-3):
        self.my_device = my_device

        # karpathy parameters
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embd = n_embed
        self.hidden_size = self.n_embd * 4

        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = 0.1
        self.head_size = self.n_embd // self.n_head

        self.max_iters = 15000
        self.eval_interval = 100
        self.learning_rate = learning_rate
        self.eval_iters = 200
        self.eval_interval = 100
        self.token_codec = TokenCodec()
        self.vocab_size = self.token_codec.vocab_size

        # self.bottleneck_dim = params.bottleneck_dim
        # self.n_heads = params.n_heads
        # self.n_enc_layers = params.n_enc_layers
        # self.n_dec_layers = params.n_dec_layers
        # self.dropout = params.dropout
        #
        # assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'