import torch
import torch.multiprocessing as mp
import random

import time
import datetime
import src
from base_model_runners import GpuRunnner
from envs import build_env
from ga_model_runner import GAModelRunnner
from src.slurm import init_distributed_mode
from src.utils import initialize_exp
from src.performance_utils import timeit
from t2.realtime_trainer import RealtimeTrainer
from t2.transformer import build_transformer
from t2.sentimental_transformer import build_sentimental_transformer
from t2.utils import get_parser, join_sai
from ga.ga import GA, TargetStringEvaluator, XY

argv = [
    '--exp_name', 'first_train',
    '--tasks', 'add_dataset',
    '--n_enc_layers', '64',
    '--n_heads', '4',
    '--sinusoidal_embeddings', 'false',
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


def model_runner_factory(gpu_num, model_num, params, env):
    return GAModelRunnner(gpu_num, model_num, params, env)


def run_gpu(number_of_iterations, gpu_num, models_per_gpu, params, env):
    gpu_runner = GpuRunnner(gpu_num, models_per_gpu, params, env, model_runner_factory)

    gpu_runner.iterate(number_of_iterations)


if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []
    number_of_gpus = 1
    models_per_gpu = 10
    number_of_iterations = 101
    # seems like multi gpu may not work???
    for gpu_num in range(number_of_gpus):
        p = mp.Process(target=run_gpu, args=(number_of_iterations, gpu_num, models_per_gpu, params, env))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
