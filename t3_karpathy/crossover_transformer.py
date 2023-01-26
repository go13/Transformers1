import torch

from ga_t3.accumulative_trainer import AbstractAccumulativeTrainer
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.transformer_runner import CrossoverRunner


class CrossoverAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: CrossoverRunner = CrossoverRunner(config)
        self.data_x1 = []
        self.data_x2 = []
        self.data_y = []
        self.data_child_f = []
        self.data_dict = dict()
    #
    # def get_fitness_histogram(self):
    #     f_hist = dict()
    #     for x1, x2, x3 in zip(self.data_x1, self.data_x2, self.data_x3):
    #         num = self.data_dict[x1]
    #         f = int(y)
    #         if f in f_hist:
    #             f_hist[f] += num
    #         else:
    #             f_hist[f] = num
    #     return f_hist

    # def get_xy_histogram(self):
    #     xy_hist = dict()
    #     i = 0
    #     for x, y in self.data_dict.items():
    #         xy_hist[i] = y
    #         i += 1
    #     return xy_hist

    def get_batch(self):
        ix = torch.randint(len(self.data_x1), (self.config.batch_size,))
        x1 = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x1[i])) for i in ix])
        x2 = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x2[i])) for i in ix])
        y = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_y[i])) for i in ix])
        f = torch.stack([torch.tensor(self.data_child_f[i]) for i in ix])
        x1, x2, y, f = x1.to(self.config.my_device), x2.to(self.config.my_device), y.to(self.config.my_device), f.to(self.config.my_device)
        return x1, x2, y, f

    def add_sample(self, x1, x2, y, child_f):
        key = (x1, x2)
        if key in self.data_dict:
            self.data_dict[key] += 1
            return self.data_dict[key]

        self.data_x1 += [x1]
        self.data_x2 += [x2]
        self.data_y += [y]
        self.data_child_f += [child_f]

        self.data_dict[key] = 1

        return 1

    def predict_list(self, lst1, lst2):
        lst1 = [self.config.token_codec.encode(x) for x in lst1]
        lst2 = [self.config.token_codec.encode(x) for x in lst2]

        x1 = torch.tensor(lst1).to(self.config.my_device)
        x2 = torch.tensor(lst2).to(self.config.my_device)

        out = self.runner.generate(x1, x2)
        out = out.tolist()

        return [self.config.token_codec.decode(o) for o in out]

    def predict(self, x1, x2):
        return self.predict_list([x1], [x2])[0]

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x1, x2, y, f = self.get_batch()
            o, loss = self.runner.learn(x1, x2, y, f)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x1)