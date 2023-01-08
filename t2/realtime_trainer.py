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

    def learn_detailed(self, x1, x2, y):
        x1 = join_sai(x1)
        x2 = join_sai(x2)
        y = join_sai(y)
        self.training_queue.append((x1.split(), x2.split(), y.split()))

        if len(self.training_queue) == self.params.batch_size:
            (x1, x1_len), (x2, x2_len), (y1, y_len), _ = self.collate_fn(self.training_queue)

            loss = self._learn_detailed(x1, x1_len, x2, x2_len, y1, y_len)

            self.training_queue.clear()

            return True, loss
        return False, None

    def act(self, xx1, xx2):
        q = []
        for x1, x2 in zip(xx1, xx2):
            q.append((x1.split(), x2.split(), x2.split()))

        (x1, len1), (x2, len2), _, _, _ = self.collate_fn(q)

        return self._act_detailed(x1, len1, x2, len2)

    def single_act(self, inp1, inp2):
        inp1 = join_sai(inp1)
        inp2 = join_sai(inp2)
        lst1 = []
        lst2 = []
        for _ in range(self.params.batch_size):
            x1 = inp1
            lst1 += [x1]

            x2 = inp2
            lst2 += [x2]

        return self.act(lst1, lst2)[0]
