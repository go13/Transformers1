import torch

from t3_karpathy.commons.commons import BaseTransformerConfig, AbstractDataLoader


class GenericDataloader(AbstractDataLoader):
    def __init__(self, config: BaseTransformerConfig, data):
        super().__init__(config)
        self.data = data.to(config.precision).to(self.config.my_device)
        self.config = config

        n = int(0.9 * len(self.data))  # first 90% will be trained, rest val
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

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