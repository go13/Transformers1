class TransformerConfig:
    STORE_OUTPUTS = False

    def __init__(self, params, id2word, with_output):
        # self.is_encoder = is_encoder
        # self.is_decoder = not is_encoder
        self.my_device = params.my_device
        self.with_output = with_output
        self.batch_size = params.batch_size
        self.share_inout_emb = params.share_inout_emb
        self.input_seq_length = params.input_seq_length
        self.sinusoidal_embeddings = params.sinusoidal_embeddings

        # dictionary
        self.n_words = params.n_words
        self.eos_index = params.eos_index
        self.pad_index = params.pad_index
        self.id2word = id2word
        assert len(self.id2word) == self.n_words

        # model parameters
        self.dim = params.emb_dim  # 512 by default
        self.hidden_dim = self.dim * 4  # 2048 by default
        self.bottleneck_dim = params.bottleneck_dim
        self.n_heads = params.n_heads  # 8 by default
        self.n_enc_layers = params.n_enc_layers
        self.n_dec_layers = params.n_dec_layers
        self.dropout = params.dropout
        self.attention_dropout = params.attention_dropout
        assert self.dim % self.n_heads == 0, 'transformer dim must be a multiple of n_heads'