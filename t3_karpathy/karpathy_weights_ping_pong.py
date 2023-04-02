import time

import torch

from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig

from t3_karpathy.karpathy_transformer import KarpathyRunner

torch.manual_seed(1337)

config = TransformerConfig()
dataloader = GptNanoDataloader(config)

print("Training runner 1")
runner1 = KarpathyRunner(config)
runner1.train_eval(5, dataloader.get_train_batch, dataloader.get_val_batch)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(dataloader.decode(runner1.generate(context, max_new_tokens=2000)[0].tolist()))


print("Training runner 2")
runner2 = KarpathyRunner(config)
runner2.train_eval(5, dataloader.get_train_batch, dataloader.get_val_batch)
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(dataloader.decode(runner2.generate(context, max_new_tokens=2000)[0].tolist()))

# Now let's play ping pong

n_iter = 1000
t = time.time()
for i in range(n_iter):
    w = runner1.get_weights()
    runner2.set_weights(w)
t = time.time() - t
print(f"Time to run {n_iter} iterations: {t}, {n_iter / t} it/s, {t / n_iter * 1000} ms/it")
