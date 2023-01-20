import torch
import torch.multiprocessing as mp

import src
from base_model_runners import GpuRunnner
from envs import build_env
from ga_model_runner import GAModelRunnner
from src.slurm import init_distributed_mode
from src.utils import initialize_exp
from t2.utils import get_parser

argv = [
    '--exp_name', 'first_train',
    '--tasks', 'add_dataset',
    '--n_enc_layers', '4',
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

#TODO: put xy1 and 2 into encoder and decoder should iterate and generate a child
# as there is a triangular mask that is used in the decoder

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


def run_gpu(gpu_num, params, env):
    gpu_runner = GpuRunnner(gpu_num, params, env, model_runner_factory)

    gpu_runner.iterate(params.number_of_iterations)


if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []

    params.number_of_gpus = 1
    params.models_per_gpu = 1
    params.number_of_iterations = 1000
    params.log_ga_into_file = False
    params.verbose = True

    params.ga_use_random_exchange = False
    params.ga_population_size = 20

    params.use_neural_crossover = False
    params.neural_crossover_iteration_threshold = 200

    params.exchange_best_every_n_iterations = 1
    params.select_best_of_group = 5
    params.distribute_best = 5

    # seems like multi gpu may not work???
    for gpu_num in range(params.number_of_gpus):
        p = mp.Process(target=run_gpu, args=(gpu_num, params, env))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
