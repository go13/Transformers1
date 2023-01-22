import argparse

import torch.multiprocessing as mp

from ga_model_runner import GAModelRunnner
from ga_t3.base_model_runner import GpuRunnner


argv = [
]

parser = argparse.ArgumentParser(description="Language transfer")

# main parameters
parser.add_argument("--number_of_gpus", type=int, default=1)
parser.add_argument("--models_per_gpu", type=int, default=1)
parser.add_argument("--number_of_iterations", type=int, default=200)
parser.add_argument("--log_ga_into_file", type=bool, default=True)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--ga_use_random_exchange", type=bool, default=False)
parser.add_argument("--ga_population_size", type=int, default=20)
parser.add_argument("--use_neural_crossover", type=bool, default=False,)
parser.add_argument("--neural_crossover_iteration_threshold", type=int, default=200)
parser.add_argument("--exchange_best_every_n_iterations", type=int, default=1)
parser.add_argument("--select_best_of_group", type=int, default=5)
parser.add_argument("--distribute_best", type=int, default=10)

params = parser.parse_args(argv)
print(params)


def model_runner_factory(gpu_num, model_num, params):
    return GAModelRunnner(gpu_num, model_num, params)


def run_gpu(gpu_num, params):
    gpu_runner = GpuRunnner(gpu_num, params, model_runner_factory)

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
        p = mp.Process(target=run_gpu, args=(gpu_num, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()