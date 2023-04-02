import torch

from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.sentimental_transformer import SentimentalRunner

torch.manual_seed(1337)

config = TransformerConfig()

bs = 16
XB = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8] for i in range(bs)]).to('cuda')
XX = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8] for i in range(1)]).to('cuda')


def get_batch():
    x = XB
    y = torch.tensor([0.33744669 for i in range(bs)])
    x, y = x.to('cuda'), y.to('cuda')
    return x, y


print("Training runner 1")
runner1 = SentimentalRunner(config)

runner1.train_eval(15000, get_batch, get_batch)

print(runner1.forward(XX))
