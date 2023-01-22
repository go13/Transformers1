class TransformerConfig:

    def __init__(self, my_device='cuda'):
        self.my_device = my_device

        # karpathy parameters
        self.batch_size = 16 # how many independent sequences will we process in parallel?
        self.block_size = 32 # what is the maximum context length for predictions?
        self.n_embd = 64
        self.hidden_size = self.n_embd * 4

        self.n_head = 4
        self.n_layer = 4
        self.dropout = 0.1
        self.head_size = self.n_embd // self.n_head

        self.max_iters = 15000
        self.eval_interval = 100
        self.learning_rate = 1e-3
        self.eval_iters = 200
        self.vocab_size = 65
        self.eval_interval = 100

        # self.bottleneck_dim = params.bottleneck_dim
        # self.n_heads = params.n_heads
        # self.n_enc_layers = params.n_enc_layers
        # self.n_dec_layers = params.n_dec_layers
        # self.dropout = params.dropout
        #
        # assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'