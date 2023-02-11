import torch

from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig

from t3_karpathy.enhanced_karpathy_transformer import KarpathyRunner

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

config = TransformerConfig(n_embed=64, n_head=4, n_layer=4, batch_size=32) # batch_size=1024, block_size=128, n_embed=64, n_head=4, n_layer=8)
runner = KarpathyRunner(config)
dataloader = GptNanoDataloader(config)

runner.train_iterate(5000, dataloader.get_train_batch, dataloader.get_val_batch)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(dataloader.token_codec.decode(runner.generate(context, max_new_tokens=2000)[0].tolist()))