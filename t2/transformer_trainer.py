import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch

from src.optim import get_optimizer
from src.utils import to_cuda, words2string, ids2words
from t2.encoder_transformer import EncoderTransformer
from torch.nn import functional as F
logger = getLogger()


class TransformerTrainer(object):

    def __init__(self, env, params):
        # modules / params
        self.modules = EncoderTransformer.build_transformer(env, params)
        self.params = params
        self.env = env
        self.my_device = params.my_device

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        assert params.amp < 0

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [('processed_e', 0)] +
            [('processed_w', 0)] +
            sum([
                [(x, []), (f'{x}-AVG-STOP-PROBS', [])]
                for x in env.TRAINING_TASKS
            ], [])
        )
        self.last_time = time.time()

        self.training_queue = []
        self.learning_queue = []
        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1_000_000_000)

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters['model'] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()


    def learn(self, x1, len1, y):

        transformer = self.modules['transformer']
        transformer.train()

        loss = self.fwd2(transformer, x1, len1, y)

        self.optimize(loss)

        if False:
            result = self.log_in_out(bs, output, x1, len1, x2)

        # print(f"learning: device={self.my_device}, loss={loss.item()}")

        logger.info(f"learning: device={self.my_device}, loss={loss.item()}")

        # e = abs(input_size - str_diff(pred, src1)) / input_size

        return loss

    @torch.no_grad()
    def estimate_loss(self):
        from torch.nn import functional as F

        transformer = self.modules['transformer']
        transformer.eval()
        out = {}
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                x, y, len1 = get_batch(split)

                loss = self.fwd2(transformer, x, len1, y)
                # print(decode(transformer_trainer.generate(xb, max_new_tokens=1000)[0].tolist()))

                losses[k] = loss.item()
            out[split] = losses.mean()
        transformer.train()
        return out

    def fwd2(self, transformer, x1, len1, targets):
        # targets = to_cuda(self.my_device, targets)
        x1, len1 = to_cuda(self.my_device, x1, len1)
        logits = transformer('fwd', x1=x1, len1=len1)
        # logits = output.max(1)[1].reshape(-1, bs)

        logits = logits.view(-1, self.params.emb_token_size)
        targets = targets.reshape(-1)
        loss = F.cross_entropy(logits, targets)
        return loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits = self.fwd(idx_cond)
            # focus only on the last time step
            logits = logits[-1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    def fwd(self, x1):
        len1 = torch.full((1, ), x1.shape[0], device='cuda')

        x1, len1 = to_cuda(self.my_device, x1, len1)

        logits = transformer('fwd', x1=x1, len1=len1)
        logits = logits.view(-1, self.params.emb_token_size)
        return logits

    def get_pred_mask(self, len1):
        alen = torch.arange(len1.max(), dtype=torch.long, device=self.my_device)
        # pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        pred_mask = alen[:, None] <= len1[None] - 1  # let's predict all n words as it is a generator only
        return pred_mask

    def get_transformer(self):
        return self.modules['transformer']


import torch

import src
from envs import build_env
from t2.utils import get_parser

argv = [
    '--exp_name', 'first_train',
    '--tasks', 'add_dataset',
    '--n_enc_layers', '4',
    '--n_heads', '4',
    '--sinusoidal_embeddings', 'false',
    '--emb_token_size', '65',
    '--num_workers', '4',
    '--batch_size', '16',
    '--dropout', '0.1',
    '--attention_dropout', '0.1',
    '--emb_dim', '64',
    '--bottleneck_dim', '64',
    '--nn_output', '1',
    '--input_seq_length', '32',
    '--share_inout_emb', 'false',
    '--eval_onl', '0',
    '--save_periodic', '0',
    '--epoch_size', '10000',
]

#TODO: put xy1 and 2 into encoder and decoder should iterate and generate a child
# as there is a triangular mask that is used in the decoder

parser = get_parser()

params = parser.parse_args(argv)
params.my_device = 'cuda'

src.utils.CUDA = not params.cpu

env = build_env(params)

print(params)

bs = params.batch_size

# CPU / CUDA
if params.cpu:
    assert not params.multi_gpu
else:
    assert torch.cuda.is_available()

transformer_trainer = TransformerTrainer(env, params)
transformer = transformer_trainer.get_transformer()


# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

# torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('F://workspace//ai//Transformers1//input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"vocab_size={vocab_size}")
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xl=[data[i:i + block_size] for i in ix]
    x = torch.stack(xl)
    yl=[data[i + 1:i + block_size + 1] for i in ix]
    y = torch.stack(yl)

    # print(decode(xl[0].tolist()))
    # print(decode(yl[0].tolist()))

    len1=torch.full((batch_size, ), block_size, device='cuda')
    x, y = x.to(device), y.to(device)
    return x, y, len1

def get_constant_batch(split, offset):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    xl=[data[offset:offset + block_size] for i in range(batch_size)]
    x = torch.stack(xl)
    yl=[data[offset + 1:offset + block_size + 1] for i in range(batch_size)]
    y = torch.stack(yl)

    len1=torch.full((batch_size, ), block_size, device='cuda')
    x, y = x.to(device), y.to(device)
    return x, y, len1


for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = transformer_trainer.estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x, y, len1 = get_batch('train')

    loss = transformer_trainer.learn(x1=x, len1=len1, y=y)

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(transformer_trainer.generate(context, max_new_tokens=2000)[0].tolist()))
