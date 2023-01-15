import torch
import torch.multiprocessing as mp
import random

import time
import datetime
import src
from envs import build_env
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

class ModelRunnner(object):
    def __init__(self, gpu_num, model_num, params):
        self.gpu_num = gpu_num
        self.model_num = model_num
        self.params = params

        current_date_time = time.strftime("%H-%M-%S", time.localtime())
        self.log_file = open(f"evolution-{gpu_num}-{current_date_time}.txt", "w")

        self.crossover_transformer = build_transformer(env, params)
        self.crossover_trainer = RealtimeTrainer(self.crossover_transformer, env, params)

        self.training_set = set()

        self.ga = GA(TargetStringEvaluator())
        self.ga.evaluate()
        self.ga.sort_population()

        self.start_time = time.time()

    @timeit
    def step(self, iteration_num, gpu_num, params):

        # sentimental_transformer = build_sentimental_transformer(env, params)
        # sentimental_trainer = RealtimeTrainer(sentimental_transformer, env, params)

        tm = time.time()
        ga = self.ga

        ga.print_population()

        if ga.iteration > 200 or True: #random.random() > 0.5 and
            children, families = neural_crossover(ga, params, self.crossover_trainer)
        else:
            children, families = ga.crossover()

        for a, b, c in families:
            self.log_file.write(f"crossover,{iteration_num},{a.data},{b.data},{c.data}\n")

        # for xy in ga.population:
        #     print(crossover_trainer.modules['transformer'].state_dict())

        children = ga.mutate(children)

        for c in children:
            self.log_file.write(f"mutated,{iteration_num},{c.data}\n")

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        for c in ga.population:
            self.log_file.write(f"evaluated,{iteration_num},{c.f},{c.data}\n")

        # learn crossover result
        for a, b, c in families:
            df = (c.f - max(a.f, b.f))
            # if df < 0:
            #     df = df * 0.001
            # for _ in range(params.batch_size):
            df = 1
            self.training_set.add((a.data, b.data, c.data, df))

        for (a, b, c, df) in  random.sample(self.training_set, min(params.batch_size * 1, len(self.training_set))):
            self.crossover_trainer.learn_accumulate(a, b, c, df)

        # sentimental_trainer.learn_accumulate()

        # model_weights = self.crossover_transformer['transformer'].state_dict()
        # model_weights = {k: v.cpu() for k, v in model_weights.items()}



        ga.iteration += 1

        tm_new = time.time()

        print(f"Time of iteration is {tm_new - tm}, it=on gpu {gpu_num}")

        self.log_file.write(f"iteration_time,{iteration_num},{tm_new - tm}\n")

        tm = tm_new

        # end_time = time.time()
        #
        # print(f"Total time taken = {end_time - start_time}")
        # print(f"Average time per iteration = {(end_time - start_time) / iterations}")




def step_all_models(iteration_num, gpu_num, runners, models_per_gpu, params):
    for r in runners:
        r.step(iteration_num, gpu_num, params)


def run_all_models_per_gpu(number_of_iterations, gpu_num, models_per_gpu, params):
    runners = [ModelRunnner(gpu_num, model_num, params) for model_num in range(models_per_gpu)]

    for iteration_num in range(number_of_iterations):
        print(f"Starting iteration {iteration_num} on gpu {gpu_num}")
        start = time.time()

        step_all_models(iteration_num, gpu_num, runners, models_per_gpu, params)

        end = time.time()
        print(f"Ended iteration {iteration_num} on gpu {gpu_num}, taken = {end - start}, time/iteration = {(end - start) / models_per_gpu}, model_num={models_per_gpu}")

def neural_crossover(ga, params, trainer):
    children = []
    families = []
    p1, p2 = ga.select_random_parents(params.batch_size)
    pp1 = xy_to_data(p1)
    pp2 = xy_to_data(p2)
    children_data = trainer.act(pp1, pp2)

    for p1, p2, ch_data in zip(p1, p2, children_data):
        ch = XY('', ch_data)
        children += [ch]
        families += [(p1, p2, ch)]

    # pre_eval = pre_evaluate(children, None)

    # pre_eval, children, families = zip(pre_eval, children, families).sort()

    children = children[:ga.new_size]
    families = families[:ga.new_size]

    return children, families


def pre_evaluate(children, trainer):
    ch_data = [xy_to_data(ch) for ch in children]
    ch_f = trainer.act(ch_data, ch_data)
    return ch_f

def xy_to_data(p1):
    pp1 = [p.data for p in p1]
    return pp1


if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []
    number_of_gpus = 1
    models_per_gpu = 10
    number_of_iterations = 100
    # seems like multi gpu may not work???
    for gpu_num in range(number_of_gpus):
        params.my_device = 'cuda:' + str(gpu_num)

        p = mp.Process(target=run_all_models_per_gpu, args=(number_of_iterations, gpu_num, models_per_gpu, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
