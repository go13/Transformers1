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


def run(rank, trainer, params):
    ga = GA(TargetStringEvaluator())
    ga.evaluate()
    ga.sort_population()
    bs = trainer.params.batch_size
    for i in range(1000):
        ga.print_population()

        children, families = ga.crossover()

        children = ga.mutate(children)

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        # learn crossover result
        for a, b, c in families:
            df = (c.f - max(a.f, b.f))
            #for _ in range(bs):
            trainer.learn(a.data, b.data, c.data, df)

        ga.iteration += 1


    # A = 'KAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAE'
    # x = join_sai(A)
    #
    # iterations_number = 100
    # for epoch in range(iterations_number):
    #     for _ in range(params.batch_size):
    #         loss = trainer.learn(x, x)
    #         if loss[0]:
    #             print(f"epoch={epoch}, rank={rank}")


if __name__ == '__main__':
    print('started')

    mp.set_start_method('spawn', force=True)

    processes = []
    number_of_gpus = 2
    number_of_models = 2

    for rank in range(number_of_models):
        params.my_device = 'cuda:' + str(rank % number_of_gpus)
        trainer = RealtimeTrainer(build_transformer(env, params), env, params)

        p = mp.Process(target=run, args=(rank, trainer, params))

        processes += [p]

    for p in processes:
        p.start()

    for p in processes:
        p.join()
