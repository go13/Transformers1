import string
from logging import getLogger
import os
import io
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

logger = getLogger()

class SAIEnvironment(object):

    print("SAIEnvironment created")

    TRAINING_TASKS = {'prim_fwd', 'prim_bwd', 'prim_ibp', 'ode1', 'ode2', 'add_dataset'}

    WORD_DICTIONARY_SPECIAL = [
        '<s>', '</s>'
    ]

    WORD_DICTIONARY = WORD_DICTIONARY_SPECIAL + list(string.ascii_uppercase + string.digits)

    def __init__(self, params):

        self.words = self.WORD_DICTIONARY
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"words: {self.word2id}")

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(self.pad_index)
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1:lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        pass

    def create_train_iterator(self, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            rng=None,
            params=params,
            path=(None if data_path is None else data_path[task][0])
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=params.batch_size,
            num_workers=(params.num_workers if data_path is None or params.num_workers == 0 else 1),
            shuffle=False,
            # pin_memory=True,
            collate_fn=dataset.collate_fn
        )

    def create_test_iterator(self, data_type, task, params, data_path):
        """
        Create a dataset for this environment.
        """
        assert data_type in ['valid', 'test']
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            rng=np.random.RandomState(0),
            params=params,
            path=(None if data_path is None else data_path[task][1 if data_type == 'valid' else 2])
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=params.batch_size,
            num_workers=1,
            shuffle=False,
            collate_fn=dataset.collate_fn
        )


class EnvDataset(Dataset):

    def __init__(self, env, task, train, rng, params, path):
        super(EnvDataset).__init__()
        self.env = env
        self.rng = rng
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        assert (train is True) == (rng is None)
        assert task in SAIEnvironment.TRAINING_TASKS

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size
        self.same_nb_ops_per_batch = params.same_nb_ops_per_batch

        # generation, or reloading from file
        if path is not None:
            assert os.path.isfile(path)
            logger.info(f"Loading data from {path} ...")
            with io.open(path, mode='r', encoding='utf-8') as f:
                # either reload the entire file, or the first N lines (for the training set)
                if not train:
                    lines = [line.rstrip().split('|') for line in f]
                else:
                    lines = []
                    for i, line in enumerate(f):
                        if i == params.reload_size:
                            break
                        if i % params.n_gpu_per_node == params.local_rank:
                            lines.append(line.rstrip().split('|'))
            self.data = [xy.split('\t') for _, xy in lines]
            self.data = [xy for xy in self.data if len(xy) == 2]
            logger.info(f"Loaded {len(self.data)} records from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test (default of 5000 if no file provided)
        if self.train:
            self.size = 1 << 60
        else:
            self.size = 5000 if path is None else len(self.data)

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)

        nb_ops = [sum(int(word in self.env.WORD_DICTIONARY) for word in seq) for seq in x]

        x = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id]) for seq in y]

        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)

        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if self.rng is None:
            assert self.train is True
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.rng = np.random.RandomState([worker_id, self.global_rank, self.env_base_seed])
            logger.info(f"Initialized random generator for worker {worker_id}, with seed {[worker_id, self.global_rank, self.env_base_seed]} (base seed={self.env_base_seed}).")

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        if self.train:
            index = self.rng.randint(len(self.data))
        x, y = self.data[index]
        x = x.split()
        y = y.split()
        assert len(x) >= 1 and len(y) >= 1
        return x, y
