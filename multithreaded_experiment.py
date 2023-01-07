import torch
import torch.multiprocessing as mp

import src
from envs import build_env
from src.slurm import init_distributed_mode
from src.utils import initialize_exp
from t2.realtime_trainer import RealtimeTrainer
from t2.transformer import build_transformer
from t2.utils import get_parser, join_sai

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
    number_of_models = 8
    number_of_gpus = 2
    for i in range(number_of_models):
        params.my_device = 'cuda:' + str(i % number_of_gpus)
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
