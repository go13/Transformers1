import torch

from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig

from t3_karpathy.enhanced_karpathy_transformer import EnhancedKarpathyRunner

device = 'cuda'

torch.manual_seed(1337)

config = TransformerConfig(precision=torch.bfloat16, n_embed=64, n_head=4, n_layer=4, batch_size=32, block_size=128) # batch_size=1024, block_size=128, n_embed=64, n_head=4, n_layer=8)
runner = EnhancedKarpathyRunner(config)
dataloader = GptNanoDataloader(config)

runner.train_iterate(25000, dataloader.get_train_batch, dataloader.get_val_batch)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataloader.token_codec.decode(runner.generate(context, max_new_tokens=2000)[0].tolist()))