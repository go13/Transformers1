from t3_karpathy.compressing_transformer import CompressingAccumulativeTrainer
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.karpathy_transformer import KarpathyRunner
import torch

config = TransformerConfig()
dataloader = GptNanoDataloader(config)
device = 'cuda'

trainer1=CompressingAccumulativeTrainer(config)
#rainer1=KarpathyRunner(config)
trainer1.train_iterate(5000, dataloader.get_train_batch, dataloader.get_val_batch)

print(dataloader.token_codec.decode(trainer1.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 100)[0].tolist()))

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataloader.token_codec.decode(trainer1.generate(context, max_new_tokens=2000)[0].tolist()))