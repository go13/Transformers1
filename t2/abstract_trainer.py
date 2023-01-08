import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch

from src.optim import get_optimizer
from src.utils import to_cuda, words2string, ids2words, str_diff

logger = getLogger()


class AbstractTrainer(object):

    def __init__(self, modules, env, params):
        # modules / params
        self.modules = modules
        self.params = params
        self.env = env
        self.my_device = params.my_device

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        assert params.amp < 0

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(',') if m != '']
        for m in metrics:
            m = (m[1:], False) if m[0] == '_' else (m, True)
            self.metrics.append(m)
        self.best_metrics = {metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics}

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [('processed_e', 0)] +
            [('processed_w', 0)] +
            sum([
                [(x, []), (f'{x}-AVG-STOP-PROBS', [])]
                for x in env.TRAINING_TASKS
            ], [])
        )
        self.last_time = time.time()

        if params.env_base_seed < 0:
            params.env_base_seed = np.random.randint(1_000_000_000)

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend([(k, p) for k, p in v.named_parameters() if p.requires_grad])
        self.parameters['model'] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizers
        names = self.optimizers.keys()
        optimizers = [self.optimizers[k] for k in names]

        # regular optimization
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 20 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.4f}'.format(k.upper().replace('_', '-'), np.mean(v))
            for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = ""
        for k, v in self.optimizers.items():
            s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.4e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats['processed_e'] * 1.0 / diff,
            self.stats['processed_w'] * 1.0 / diff
        )
        self.stats['processed_e'] = 0
        self.stats['processed_w'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        x, y = zip(*elements)

        nb_ops = [sum(int(word in self.env.WORD_DICTIONARY) for word in seq) for seq in x]

        x = [self.seq2tensor(seq) for seq in x]
        y = [self.seq2tensor(seq) for seq in y]

        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)

        return (x, x_len), (y, y_len), torch.LongTensor(nb_ops)

    def seq2tensor(self, seq):
        return torch.LongTensor([self.env.word2id[w] for w in seq if w in self.env.word2id])

    def _learn(self, x1, len1, y, y_len):
        transformer = self.modules['transformer']
        transformer.train()

        x1, len1, y = to_cuda(self.my_device, x1, len1, y)

        # target words to predict
        alen = torch.arange(len1.max(), dtype=torch.long, device=self.my_device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word
        y = y[1:].masked_select(pred_mask[:-1])
        # y = x1[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len1 - 1).sum().item()

        # pred_mask = pred_mask.transpose(0, 1)

        # x1 = x1.transpose(0, 1)

        tensor = transformer('fwd', x1=x1, len1=len1)
        scores, loss = transformer('learn', tensor=tensor, pred_mask=pred_mask, y=y)

        self.optimize(loss)

        # logger.info(f"learning: loss={loss}")

        input_size = self.params.input_seq_length
        bs = self.params.batch_size
        # scores = scores.reshape(-1, bs, self.params.n_words)
        # scores = scores.transpose(0, 1)
        output = scores
        o = output.max(1)[1].reshape(-1, self.params.batch_size)
        result = []
        av_score = 0
        for i in range(bs):
            src = x1[1:len1[i] - 1, i].tolist()
            src = words2string(ids2words(self.env.id2word, src))

            pred = o[0:len1[i] - 2, i].tolist()
            pred = words2string(ids2words(self.env.id2word, pred))
            e = abs(input_size - str_diff(pred, src)) / input_size
            av_score += e
            #logger.info(f"learning: src={src}, pred={pred}, score={e}")

            result += [pred]
        av_score = av_score / bs

        logger.info(f"learning: av-score={av_score}, device={self.my_device}")

        return loss

    def _act(self, x1, len1):
        return self._act_detailed(x1, len1, x1, len1)

    def _act_detailed(self, x1, len1, x2, len2):
        transformer = self.modules['transformer']
        transformer.eval()

        # cuda
        x1, len1, x2, len2 = to_cuda(self.my_device, x1, len1, x2, len2)

        alen = torch.arange(len1.max(), dtype=torch.long, device=self.my_device)
        pred_mask = alen[:, None] < len1[None] - 1  # do not predict anything given the last target word

        # pred_mask = pred_mask.transpose(0, 1)

        bs = self.params.batch_size
        # forward / loss
        tensor = transformer('fwd', x1=x1, len1=len1, x2=x2, len2=len2)
        output = transformer('generate', tensor=tensor, pred_mask=pred_mask)
        # output = output.reshape(bs, -1, self.params.n_words)
        # i = random.randint(0, bs - 1)

        #input_size = self.params.input_seq_length
        # bs = self.params.batch_size
        # scores = scores.reshape(-1, bs, self.params.n_words)
        # scores = scores.transpose(0, 1)

        o = output.max(1)[1].reshape(-1, self.params.batch_size)
        result = []
        av_score = 0
        for i in range(bs):
            input_size = len1[i]
            src = x1[1:len1[i] - 1, i].tolist()

            pred = o[0:len1[i] - 2, i].tolist()
            e = abs(input_size - str_diff(pred, src)) / input_size
            av_score += e

            src = words2string(ids2words(self.env.id2word, src))
            pred = words2string(ids2words(self.env.id2word, pred))

            #if self.params.eval_verbose_print:

            # logger.info(f"acting: src={src}, pred={pred}, score={e}")

            result += [pred]
        av_score = av_score / bs

        # logger.info(f"acting: av-score={av_score}")

        return result
