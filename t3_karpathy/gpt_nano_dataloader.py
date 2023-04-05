import torch

from t3_karpathy.commons.commons import BaseTransformerConfig, AbstractDataLoader


class GptNanoDataloader(AbstractDataLoader):
    def __init__(self, config: BaseTransformerConfig):
        super().__init__(config)
        self.config = config
        self.token_codec = self.config.token_codec

    # data loading
    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.token_codec.train_data if split == 'train' else self.token_codec.val_data
        ix = torch.randint(len(data) - self.config.block_size, (self.config.batch_size,))
        x = torch.stack([data[i:i + self.config.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.config.block_size + 1] for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device)
        return x, y

    def get_train_batch(self):
        return self.get_batch('train')

    def get_val_batch(self):
        return self.get_batch('test')
