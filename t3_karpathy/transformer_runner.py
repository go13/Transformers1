import torch
from collections import OrderedDict

from t3_karpathy.transformer import KarpathyTransformerModel, CrossoverTransformerModel
from t3_karpathy.transformer_config import TransformerConfig


class AbstractRunner(object):
    def __init__(self, config: TransformerConfig, model):
        self.model = model.to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

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

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))


class KarpathyRunner(AbstractRunner):
    def __init__(self, config: TransformerConfig):
        super().__init__(config, KarpathyTransformerModel(config))
        pass

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)


class CrossoverRunner(object):
    def __init__(self, config: TransformerConfig):
        self.model = CrossoverTransformerModel(config).to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def forward(self, x1, x2):
        return self.model(x1, x2)

    def generate(self, x1, x2):
        return self.model.generate(x1, x2)

    def learn(self, x1, x2, y, f):
        self.model.train()
        out, loss = self.model.forward_vs_target(x1, x2, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward(gradient=f.sum())
        self.optimizer.step()
        return out, loss

    @torch.no_grad()
    def evaluate(self, get_batch, eval_iters):
        self.model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x1, x2, y = get_batch()
            logits, loss = self.model.forward_vs_target(x1, x2, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                train_losses = self.evaluate(get_train_batch, self.config.eval_iters)
                val_losses = self.evaluate(get_val_batch, self.config.eval_iters)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

            x1, x2, y, f = get_train_batch()

            logits, loss = self.learn(x1, x2, y, f)

            self.current_iteration += 1

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, new_state_dict):
        self.model.load_state_dict(OrderedDict(new_state_dict))
