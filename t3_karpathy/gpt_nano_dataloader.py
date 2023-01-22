import torch

from t3_karpathy.transformer_config import TransformerConfig


class GptNanoDataloader(object):

    def __init__(self, config: TransformerConfig):
        self.config = config

        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open('../input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.encode = lambda s: [self.stoi[c] for c in s]  # encoder: take a string, output a list of integers
        self.decode = lambda l: ''.join([self.itos[i] for i in l])  # decoder: take a list of integers, output a string

        # Train and test splits
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device)
        return x, y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')

