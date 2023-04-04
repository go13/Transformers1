import torch

from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig

from t3_karpathy.enhanced_karpathy_transformer import EnhancedKarpathyRunner

device = 'cuda'

torch.manual_seed(1337)

config = TransformerConfig(precision=torch.bfloat16, batch_size=32, block_size=128, n_embed=64, n_head=4, n_layer=8)
dataloader = GptNanoDataloader(config)
runner = EnhancedKarpathyRunner(config, dataloader)

runner.train_iterate_n(25000)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataloader.token_codec.decode(runner.generate(context, max_new_tokens=2000)[0].tolist()))