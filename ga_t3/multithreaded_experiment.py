import argparse

import torch.multiprocessing as mp

from ga_model_runner import GAModelRunner
from ga_t3.base_model_runner import GpuRunnner

parser = argparse.ArgumentParser(description="Language transfer")

# main parameters
parser.add_argument("--number_of_gpus", type=int, default=1)
parser.add_argument("--models_per_gpu", type=int, default=1)

parser.add_argument("--ga_population_size", type=int, default=20)
parser.add_argument("--number_of_iterations", type=int, default=5000)
parser.add_argument("--ga_use_random_exchange", type=bool, default=False)
parser.add_argument("--ga_mutation_p", type=float, default=0.1)

parser.add_argument("--use_neural_autoencoder", type=bool, default=True)
parser.add_argument("--use_neural_autoencoder_iteration_start", type=int, default=0)

parser.add_argument("--use_neural_crossover", type=bool, default=False)
parser.add_argument("--neural_crossover_iteration_start", type=int, default=0)
parser.add_argument("--neural_crossover_regular_crossover_prob", type=float, default=0)
parser.add_argument("--neural_crossover_iterations_per_ga_iteration", type=int, default=1)

parser.add_argument("--use_evolve_transformer", type=bool, default=False)

parser.add_argument("--use_neural_estimator", type=bool, default=False)
parser.add_argument("--neural_estimator_iteration_start", type=int, default=0)
parser.add_argument("--ga_neural_estimator_iterations_per_ga_iteration", type=int, default=1)
parser.add_argument("--ga_generate_only_unique_xy", type=bool, default=True)

parser.add_argument("--log_ga_into_file", type=bool, default=False)
parser.add_argument("--verbose", type=bool, default=True)

parser.add_argument("--exchange_best_between_gpus", type=bool, default=False)
parser.add_argument("--exchange_best_every_n_iterations", type=int, default=1)
parser.add_argument("--select_best_of_group", type=int, default=5)
parser.add_argument("--distribute_best", type=int, default=5)


params = parser.parse_args([])
print(params)


def model_runner_factory(gpu_num, model_num, params):
    return GAModelRunner(gpu_num, model_num, params)


def run_gpu(gpu_num, params):
    gpu_runner = GpuRunnner(gpu_num, params, model_runner_factory)

    gpu_runner.iterate(params.number_of_iterations)


if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []

    # seems like multi gpu may not work???
    for gpu_num in range(params.number_of_gpus):
        p = mp.Process(target=run_gpu, args=(gpu_num, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
