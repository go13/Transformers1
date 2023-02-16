from t3_karpathy.compressing_transformer import CompressingAccumulativeTrainer
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
import torch

config = TransformerConfig(batch_size=128, block_size=32, n_embed=256, n_head=4, n_layer=2)
dataloader = GptNanoDataloader(config)

trainer1=CompressingAccumulativeTrainer(config)
trainer1.train_iterate(20000, dataloader.get_train_batch, dataloader.get_val_batch)

print(dataloader.token_codec.decode(trainer1.generate(torch.zeros((1, 1), dtype=torch.long, device=config.my_device), 100)[0].tolist()))

context = torch.zeros((1, 1), dtype=torch.long, device=config.my_device)
print(dataloader.token_codec.decode(trainer1.generate(context, max_new_tokens=2000)[0].tolist()))