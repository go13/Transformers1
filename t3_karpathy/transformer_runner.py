import torch

from t3_karpathy.transformer import KarpathyTransformerModel
from t3_karpathy.transformer_config import TransformerConfig


class KarpathyRunner(object):
    def __init__(self, config: TransformerConfig):
        self.model = KarpathyTransformerModel(config).to(config.my_device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.current_iteration = 0

        print(sum(p.numel() for p in self.model.parameters()) / 1e6, 'M parameters')

    def generate(self, context, max_new_tokens):
        return self.model.generate(context, max_new_tokens)

    def forward(self, x):
        return self.model(x)

    def learn(self, x, y):
        self.model.train()
        logits, loss = self.model.forward_vs_target(x, y)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return logits, loss

    @torch.no_grad()
    def evaluate(self, get_batch):
        self.model.eval()
        losses = torch.zeros(self.config.eval_iters)
        for k in range(self.config.eval_iters):
            x, y = get_batch()
            logits, loss = self.model.forward_vs_target(x, y)
            losses[k] = loss.item()
        return losses.mean()

    def train_iterate(self, n_iter, get_train_batch, get_val_batch):
        for _ in range(n_iter):
            if self.current_iteration % self.config.eval_interval == 0:
                train_losses = self.evaluate(get_train_batch)
                val_losses = self.evaluate(get_val_batch)
                print(f"step {self.current_iteration}: train loss {train_losses:.4f}, val loss {val_losses:.4f}")

            x, y = get_train_batch()

            logits, loss = self.learn(x, y)

            self.current_iteration += 1

