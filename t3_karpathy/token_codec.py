import string

import torch


class TokenCodec(object):

    def __init__(self):
        # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
        with open('F:\\workspace\\ai\\Transformers1\\input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        # text = ""
        # here are all the unique characters that occur in this text
        self.vocab = ''.join(set(text + string.ascii_letters + string.digits))
        #self.vocab = ''.join(set((string.ascii_letters + string.digits).upper()))
        self.chars = sorted(list(self.vocab))
        self.vocab_size = len(self.chars)
        # create a mapping from characters to integers
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        # self.encode = lambda s: [self.stoi[c] for c in s]  # encoder: take a string, output a list of integers
        # self.decode = lambda l: ''.join([self.itos[i] for i in l])  # decoder: take a list of integers, output a string

        # Train and test splits
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9 * len(self.data))  # first 90% will be train, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])