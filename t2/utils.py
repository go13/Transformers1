import abc
import argparse
import os

import torch.nn as nn

from envs import ENVS
from src.utils import bool_flag

N_MAX_POSITIONS = 4096  # maximum input sequence length


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="dmytro_job",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding layer size")
    parser.add_argument("--emb_token_size", type=int, default=65, help="Embedding Dict size")
    parser.add_argument("--bottleneck_dim", type=int, default=8, help="Encoded vector size")

    parser.add_argument("--nn_output", type=int, default=0,
                        help="NN as an output")
    parser.add_argument("--input_seq_length", type=int, default=13,
                        help="input seq length")
    parser.add_argument("--n_enc_layers", type=int, default=4,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=4,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")

    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=0,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum sequences length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")
    parser.add_argument("--same_nb_ops_per_batch", type=bool_flag, default=False,
                        help="Generate sequences with the same number of operators in batches.")


    parser.add_argument("--reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")

    # environment parameters
    # parser.add_argument("--env_name", type=str, default="char_sp",
    #                     help="Environment name")
    parser.add_argument("--env_name", type=str, default="sai",
                        help="Environment name")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="",
                        help="Tasks")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=False,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    return parser


class DispatchingModule(nn.Module):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super().__init__()

        # self.modules = {}

    def forward(self, mode, **kwargs):
        """
        Forward function with different forward modes.
        ### Small hack to handle PyTorch distributed.
        """
        if mode == 'fwd':
            return self.fwd(**kwargs)
        if mode == 'fwd_encode':
            return self.fwd_encode(**kwargs)
        if mode == 'fwd_decode':
            return self.fwd_decode(**kwargs)
        elif mode == 'learn':
            return self.learn(**kwargs)
        elif mode == 'generate':
            return self.generate(**kwargs)
        else:
            raise Exception("Unknown mode: %s" % mode)

    @abc.abstractmethod
    def fwd(self, **kwargs):
        """Method documentation"""
        return__metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fwd_encode(self, **kwargs):
        """Method documentation"""
        return__metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def fwd_decode(self, **kwargs):
        """Method documentation"""
        return__metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learn(self, **kwargs):
        """Method documentation"""
        return__metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def generate(self, **kwargs):
        """Method documentation"""
        return__metaclass__ = abc.ABCMeta


def there_are_invalid_chars(env, t):
    brk = False
    for c in t:
        for j in env.WORD_DICTIONARY_SPECIAL:
            if c == j:
                brk = True
                break
        if brk:
            break
    return brk


def to_sai_str(p):
    return join_sai(p.data)


def join_sai(data):
    return ' '.join(data)


def check_model_params(params):
    """
    Check models parameters.
    """
    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # reload a pretrained model
    if params.reload_model != '':
        assert os.path.isfile(params.reload_model)
