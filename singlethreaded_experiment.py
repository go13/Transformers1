import math
import random
import string
import sys
import time
import numpy as np
import torch

import matplotlib.pyplot as plt
import src
from envs import build_env
# from src.evaluator import Evaluator
from src.slurm import init_signal_handler, init_distributed_mode
from t2.realtime_trainer import RealtimeTrainer
from src.utils import initialize_exp, str_diff
from t2.utils import get_parser, join_sai, to_sai_str

from t2.transformer import build_transformer

argv = [
    '--exp_name', 'first_train',
    '--tasks', 'add_dataset',
    '--n_enc_layers', '16',
    '--n_heads', '4',
    '--sinusoidal_embeddings', 'true',
    '--num_workers', '4',
    '--eval_onl', '0',
    '--save_periodic', '0',
    '--epoch_size', '10000',
    '--batch_size', '32',
    '--dropout', '0.1',
    '--attention_dropout', '0.1',
    '--emb_dim', '64',
    '--bottleneck_dim', '64',
    '--nn_output', '1',
    '--input_seq_length', '47',
    '--share_inout_emb', 'false'
]

parser = get_parser()

params = parser.parse_args(argv)
params.my_device = 'cuda'

print(params)

bs = params.batch_size
# dim = params.emb_dim

init_distributed_mode(params)
logger = initialize_exp(params)

# CPU / CUDA
if params.cpu:
    assert not params.multi_gpu
else:
    assert torch.cuda.is_available()

src.utils.CUDA = not params.cpu

env = build_env(params)

modules = build_transformer(env, params)
trainer = RealtimeTrainer(modules, env, params)

A = 'KAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE'
a = join_sai(A)
print(trainer.single_act_detailed(a, a))

# A = 'KAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEKAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE'
# a = join_sai(A)
# print(act(a, trainer))