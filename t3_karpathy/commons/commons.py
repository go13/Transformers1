from collections import OrderedDict

import torch
import time
from torch import nn as nn


def dict_weights_to_vector(w):
    w = [v for v in w.values()]
    w = torch.cat([v.flatten() for v in w])
    return w


def min_max_scaling(tensor):
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())


def extract_sliding_windows(tensor, window_size):
    windows = tensor.unfold(1, window_size, 1)
    return windows


def z_scale(tensor, dim=1):
    mean = torch.mean(tensor, dim=dim, keepdim=True)
    std = torch.std(tensor, dim=dim, keepdim=True)
    return (tensor - mean) / (std + 1e-7)


def diff(prices, dim=1):
    return torch.diff(prices, dim=dim)


def standardize(data, dim=-1):
    mean = torch.mean(data, dim=dim, keepdim=True)
    std = torch.std(data, dim=dim, keepdim=True)
    data_standardized = (data - mean) / (std + 1e-7)
    return data_standardized



class BaseTransformerConfig:

    def __init__(self, my_device='cuda', precision=torch.float32, batch_size=64, block_size=32, n_embed=64, n_head=4,
                 n_layer=4, learning_rate=1e-3):
        self.my_device = my_device

        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.hidden_size = self.n_embed * 4

        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = 0.1
        self.head_size = self.n_embed // self.n_head

        self.max_iters = 15000
        self.eval_interval = 500
        self.learning_rate = learning_rate
        self.eval_iters = 200

        self.norm_eps: float = 1e-5  # llma
        self.max_seq_len: int = 2048  # llma
        self.multiple_of: int = 256  # llma
        self.precision = precision

        self.save_model_periodically_every_n_iterations = 1000


class AbstractDataLoader(object):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()
        self.config = config

    def get_batch(self, split):
        raise NotImplementedError()

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')


class AbstractCodec(object):
    def __init__(self):
        pass

    def encode(self, s):
        raise NotImplementedError()

    def decode_into(self, src, dst):
        raise NotImplementedError()

    def decode(self, l):
        raise NotImplementedError()


class AbstractAccumulativeTrainer(object):
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.loss_hist = []

    def get_loss_history(self):
        return self.loss_hist


class AbstractRunner(object):
    def __init__(self, config: BaseTransformerConfig, model: nn.Module, data_loader: AbstractDataLoader):
        self.model = model.to(config.my_device, dtype=config.precision)
        self.parameters = self.model.parameters()
        # self.model = torch.compile(model, mode="max-autotune", backend="cudagraphs") # , fullgraph=True
        self.optimizer = torch.optim.AdamW(self.parameters, lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0
        self.data_loader = data_loader
        self.model_version = 1
        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x):
        return self.model(x)

    def learn(self, x, y):
        self.model.train()
        out, loss = self.model.forward_vs_target(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = 0
        for k in range(eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses += loss
        return losses / eval_iters

    def train_iterate_n(self, n_iter):
        self.train_iterate(n_iter, self.data_loader.get_train_batch, self.data_loader.get_val_batch)

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        eval_interval = self.config.eval_interval

        if self.config.eval_interval == -1:
            eval_interval = n_iter
            self.current_iteration = 1

        last_loss = 0
        t = time.time()
        for _ in range(n_iter):
            if eval_interval != -1 and self.current_iteration % eval_interval == 0:
                t_taken = time.time() - t
                train_losses = torch.sqrt(self.evaluate(get_train_batch, self.config.eval_iters))
                val_losses = torch.sqrt(self.evaluate(get_val_batch, self.config.eval_iters))
                print(
                    f"step {self.current_iteration}: rmse train loss {train_losses:.4f}, rmse val loss {val_losses:.4f}, sec/iter {t_taken / eval_interval}")

                t = time.time()

                if self.config.save_model_periodically_every_n_iterations != -1 and self.current_iteration % self.config.save_model_periodically_every_n_iterations == 0:
                    torch.save(self.model.state_dict(), f"model-{self.model_version}.pt")
                    print(f"saved model version {self.model_version}")
                    self.model_version += 1

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)
            last_loss = loss.item()
            self.current_iteration += 1

        return last_loss

    def get_weights(self):
        return self.model.state_dict()

    def get_weights_as_tensor(self):
        w = self.get_weights()
        return dict_weights_to_vector(w)

    def set_weights_as_tensor(self, new_state_tensor):
        new_state_dict = self.get_weights()
        for k, v in new_state_dict.items():
            new_state_dict[k] = new_state_tensor[:v.numel()].reshape(v.shape)
            new_state_tensor = new_state_tensor[v.numel():]
        self.set_weights(new_state_dict)

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)

    def generate(self, *args):
        raise NotImplementedError()


class TimeseriesFeedForward(nn.Module):
    def __init__(self, inp_size, hidden_size, out_size, dropout, bias=False):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size, bias=bias),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, out_size, bias=bias),
        )

    def forward(self, x):
        return self.net(x)
