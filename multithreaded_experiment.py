import torch
import torch.multiprocessing as mp

import src
from envs import build_env
from src.slurm import init_distributed_mode
from src.utils import initialize_exp
from t2.realtime_trainer import RealtimeTrainer
from t2.transformer import build_transformer
from t2.utils import get_parser, join_sai
from ga.ga import GA, TargetStringEvaluator

argv = [
    '--exp_name', 'first_train',
    '--tasks', 'add_dataset',
    '--n_enc_layers', '16',
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
    t = build_transformer(env, params)
    trainer = RealtimeTrainer(t, env, params)

    ga = GA(TargetStringEvaluator())
    ga.evaluate()
    ga.sort_population()

    for i in range(10000):
        ga.print_population()

        children, families = ga.crossover()

        children = ga.mutate(children)

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        # learn crossover result
        for a, b, c in families:
            df = (c.f - max(a.f, b.f))
            if df < 0:
                df = df * 0.001
            # for _ in range(bs):
            trainer.learn(a.data, b.data, c.data, df)
            #res = trainer.act(a.data, b.data) #TODO: fix act

        ga.iteration += 1

if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []
    number_of_gpus = 2
    number_of_models = 2

    for rank in range(number_of_models):
        params.my_device = 'cuda:' + str(rank % number_of_gpus)

        p = mp.Process(target=run, args=(rank, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
