import torch

from t3_karpathy.transformer_config import TransformerConfig
from t4.generic_dataloader import GenericDataloader

from t4.transformer import FastTransformerRunner

device = 'cuda'

torch.manual_seed(1337)

config = TransformerConfig(precision=torch.bfloat16, batch_size=32, block_size=512, n_embed=64, n_head=4, n_layer=4)

dataloader = GenericDataloader(config, None, None)
runner = FastTransformerRunner(config, dataloader)

runner.train_iterate_n(2500)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(dataloader.token_codec.decode(runner.generate(context, max_new_tokens=2000)[0].tolist()))