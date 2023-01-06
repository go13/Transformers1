import math
import random
import string
import sys
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import src
from envs import build_env
# from src.evaluator import Evaluator
from src.slurm import init_signal_handler, init_distributed_mode
from t2.realtime_trainer import RealtimeTrainer
from src.utils import initialize_exp
from t2.utils import get_parser

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


def act(inp, trainer):
    inp = join_sai(inp)
    lst = []
    for _ in range(bs):
        x = inp
        lst += [x]

    return trainer.act(lst)[0]


def to_sai_str(p):
    return join_sai(p.data)


def join_sai(data):
    return ' '.join(data)


def str_diff(s1, s2):
    diff = abs(len(s1) - len(s2))
    for i in range(min(len(s1), len(s2))):
        diff += 1 if s1[i] != s2[i] else 0
    return diff


def train(rank, trainer, params, x):
    iterations_number = 100
    for epoch in range(iterations_number):
        for _ in range(params.batch_size):
            loss = trainer.learn(x, x)
            if loss[0]:
                print(f"epoch={epoch}, rank={rank}")


if __name__ == '__main__':
    print('started')

    A = 'KAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE'
    a = join_sai(A)

    trainers = []
    number_of_models = 2
    for i in range(number_of_models):
        params.my_device = 'cuda:' + str(i)
        trainer = RealtimeTrainer(build_transformer(env, params), env, params)
        trainers += [trainer]

    kwargs = {'shuffle': True}

    kwargs.update({'num_workers': 1, 'pin_memory': True, })

    mp.set_start_method('spawn', force=True)

    processes = []
    rank = 0
    for trainer in trainers:
        p = mp.Process(target=train, args=(rank, trainer, params, a))
        p.start()
        processes.append(p)
        rank += 1

    for p in processes:
        p.join()
