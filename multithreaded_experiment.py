import torch
import torch.multiprocessing as mp
import random

import src
from envs import build_env
from src.slurm import init_distributed_mode
from src.utils import initialize_exp
from t2.realtime_trainer import RealtimeTrainer
from t2.transformer import build_transformer
from t2.sentimental_transformer import build_sentimental_transformer
from t2.utils import get_parser, join_sai
from ga.ga import GA, TargetStringEvaluator, XY

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


def run(rank, params):
    crossover_transformer = build_transformer(env, params)
    crossover_trainer = RealtimeTrainer(crossover_transformer, env, params)

    # sentimental_transformer = build_sentimental_transformer(env, params)
    # sentimental_trainer = RealtimeTrainer(sentimental_transformer, env, params)

    training_set = set()

    ga = GA(TargetStringEvaluator())
    ga.evaluate()
    ga.sort_population()

    with open(f"evolution-{rank}.txt", "w") as log_file:
        for i in range(1000):
            ga.print_population()

            if ga.iteration > 200: #random.random() > 0.5 and
                children, families = neural_crossover(ga, params, crossover_trainer)
            else:
                children, families = ga.crossover()

            for a, b, c in families:
                log_file.write("c,{i},{a},{b},{c}\n")

            children = ga.mutate(children)

            for c in children:
                log_file.write("m,{i},{c}\n")

            ga.update_bottom(children)

            ga.evaluate()
            ga.sort_population()

            # learn crossover result
            for a, b, c in families:
                df = (c.f - max(a.f, b.f))
                # if df < 0:
                #     df = df * 0.001
                # for _ in range(params.batch_size):
                df = 1
                training_set.add((a.data, b.data, c.data, df))

            for (a, b, c, df) in  random.sample(training_set, min(params.batch_size * 10, len(training_set))):
                crossover_trainer.learn_accumulate(a, b, c, df)

            # sentimental_trainer.learn_accumulate()

            ga.iteration += 1




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
    number_of_models = 30

    for rank in range(number_of_models):
        params.my_device = 'cuda:' + str(rank % number_of_gpus)

        p = mp.Process(target=run, args=(rank, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
