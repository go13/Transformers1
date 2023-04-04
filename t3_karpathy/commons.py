import torch
import time
from torch import nn as nn


class BaseTransformerConfig:

    def __init__(self, my_device='cuda', precision=torch.float32, batch_size=64, block_size=32, n_embed=64, n_head=4, n_layer=4, learning_rate=1e-2):
        self.my_device = my_device

        # karpathy parameters
        self.batch_size = batch_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.hidden_size = self.n_embed * 4

        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = 0.1
        self.head_size = self.n_embed // self.n_head

        self.max_iters = 15000
        self.eval_interval = 100
        self.learning_rate = learning_rate
        self.eval_iters = 200

        self.norm_eps: float = 1e-5  # llma
        self.max_seq_len: int = 2048  # llma
        self.multiple_of: int = 256  # llma
        self.precision = precision

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

class SentimentalFeedForward(nn.Module):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__()

        inp_size = config.n_embed * config.block_size
        hidden_size = inp_size // 2  # config.hidden_size # * config.block_size
        dropout = config.dropout
        out_size = 1

        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.Dropout(dropout),
            nn.Sigmoid(),
            nn.Linear(hidden_size, out_size),
            # FeedForward(inp_size, hidden_size, out_size, dropout),
        )

    def forward(self, x):
        return self.net(x)


class AbstractCodec(object):
    def __init__(self):
        pass

    def encode(self, s: str) -> list:
        raise NotImplementedError()

    def decode(self, l: list) -> str:
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
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate_n(self, n_iter):
        self.train_iterate(n_iter, self.data_loader.get_train_batch, self.data_loader.get_val_batch)

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        t = time.time()
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                t_taken = time.time() - t
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}, time/iter {t_taken / self.config.eval_interval}")
                t = time.time()

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))

    def generate(self, *args):
        raise NotImplementedError()

