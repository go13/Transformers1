import torch

from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.transformer_runner import SentimentalRunner, CrossoverRunner


class AbstractAccumulativeTrainer(object):
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.loss_hist = []

    def get_loss_history(self):
        return self.loss_hist


class SentimentalAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.runner: SentimentalRunner = SentimentalRunner(config)
        self.data_x = []
        self.data_y = []
        self.data_dict = dict()

    def get_fitness_histogram(self):
        f_hist = dict()
        for x, y in zip(self.data_x, self.data_y):
            num = self.data_dict[x]
            f = int(y)
            if f in f_hist:
                f_hist[f] += num
            else:
                f_hist[f] = num
        return f_hist

    def get_xy_histogram(self):
        xy_hist = dict()
        i = 0
        for x, y in self.data_dict.items():
            xy_hist[i] = y
            i += 1
        return xy_hist

    def get_batch(self):
        ix = torch.randint(len(self.data_x), (self.config.batch_size,))
        x = torch.stack([torch.tensor(self.config.token_codec.encode(self.data_x[i])) for i in ix])
        y = torch.stack([torch.tensor(self.data_y[i]) for i in ix])
        x, y = x.to(self.config.my_device), y.to(self.config.my_device)
        return x, y

    def add_sample(self, x, y):
        if x in self.data_dict:
            self.data_dict[x] += 1
            return self.data_dict[x]

        self.data_x += [x]
        self.data_y += [y]

        self.data_dict[x] = 1

        return 1

    def predict_list(self, lst):
        lst = [self.config.token_codec.encode(x) for x in lst]
        x = torch.tensor(lst).to(self.config.my_device)
        out = self.runner.forward(x)
        return out.tolist()

    def predict(self, x):
        encoded_x = self.config.token_codec.encode(x)
        x = torch.tensor(encoded_x).to(self.config.my_device)
        x = x.reshape(1, x.shape[0])
        out = self.runner.forward(x)
        return out.item()

    def train(self, n=1):
        losses = 0
        for i in range(n):
            x, y = self.get_batch()
            o, loss = self.runner.learn(x, y)
            l = loss.item()
            self.loss_hist += [l]
            losses += l
        av_loss = losses / n

        return av_loss, len(self.data_x)


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

        # x1 = x1.reshape(x1.shape[0], x1.shape[0])
        # x2 = x2.reshape(1, x2.shape[0])

        out = self.runner.generate(x1, x2)
        out = out.tolist()

        return [self.config.token_codec.decode(o) for o in out]

    def predict(self, x1, x2):
        encoded_x1 = self.config.token_codec.encode(x1)
        encoded_x2 = self.config.token_codec.encode(x2)
        x1 = torch.tensor(encoded_x1).to(self.config.my_device)
        x2 = torch.tensor(encoded_x2).to(self.config.my_device)
        x1 = x1.reshape(1, x1.shape[0])
        x2 = x2.reshape(1, x2.shape[0])
        out = self.runner.generate(x1, x2)
        out = out.reshape(-1)
        out = out.tolist()

        return self.config.token_codec.decode(out)

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