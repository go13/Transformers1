import os
import sys
from logging import getLogger
import torch

from t2.abstract_trainer import AbstractTrainer
from t2.utils import join_sai

logger = getLogger()


class RealtimeTrainer(AbstractTrainer):

    def __init__(self, modules, env, params):
        super().__init__(modules, env, params)

        self.training_queue = []
        self.data_path = None

    def learn(self, x, y):
        self.training_queue.append((x.split(), y.split()))

        if len(self.training_queue) == self.params.batch_size:
            (x1, x_len), (y1, y_len), _ = self.collate_fn(self.training_queue)

            loss = self._learn(x1, x_len, y1, y_len)

            self.training_queue.clear()

            return True, loss
        return False, None

    def learn_detailed(self, x1, x2, y):
        self.training_queue.append((x1.split(), x2.split(), y.split()))

        if len(self.training_queue) == self.params.batch_size:
            (x1, x1_len), (y1, y_len), _ = self.collate_fn(self.training_queue)
            (x2, x2_len), (y1, y_len), _ = self.collate_fn(self.training_queue)

            loss = self._learn_detailed(x1, x1_len, x2, x2_len, y1, y_len)

            self.training_queue.clear()

            return True, loss
        return False, None

    def act(self, xx):
        q = []
        for x in xx:
            q.append((x.split(), x.split()))

        (x1, len1), _, _, _ = self.collate_fn(q)

        return self._act(x1, len1)

    def act_detailed(self, xx1, xx2):
        q = []
        for x1, x2 in zip(xx1, xx2):
            q.append((x1.split(), x2.split()))

        (x1, len1), (x2, len2), _ = self.collate_fn(q)

        return self._act_detailed(x1, len1, x2, len2)

    def single_act(self, inp):
        inp = join_sai(inp)
        lst = []
        for _ in range(self.params.batch_size):
            x = inp
            lst += [x]

        return self.act(lst)[0]

    def single_act_detailed(self, inp1, inp2):
        inp1 = join_sai(inp1)
        inp2 = join_sai(inp2)
        lst1 = []
        lst2 = []
        for _ in range(self.params.batch_size):
            x1 = inp1
            lst1 += [x1]

            x2 = inp2
            lst2 += [x2]

        return self.act_detailed(lst1, lst2)[0]

    def collate_fn2(self, training_queue):
        return self.collate_fn(training_queue)