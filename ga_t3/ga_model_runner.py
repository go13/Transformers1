import time
import random
import torch
from ga.ga import GA, XY, gen_rnd_chars, crossover_string, AbstractEvaluator, TargetStringEvaluator
from ga_t3.base_model_runner import AbstractModelRunnner
from src.performance_utils import timeit
from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig
from t3_karpathy.transformer_runner import KarpathyRunner


def neural_crossover_and_mutate(xy1_weights, xy2_weights, my_device):
    new_weights = []
    for k1, _ in xy1_weights.items():
        shape = xy1_weights[k1].shape
        v1 = xy1_weights[k1].reshape(-1)
        v2 = xy2_weights[k1].reshape(-1)

        rnd1 = torch.rand(v1.shape, device=my_device)

        update = (v1 * rnd1 + (1 - rnd1) * v2)

        # mutate
        ln = len(v1)
        mutation_rate = 0.01
        num_of_ones = int(mutation_rate * ln)
        num_of_zeros = ln - num_of_ones
        ones_to_mutate = torch.ones(num_of_ones, device=my_device)
        zeros_to_mutate = torch.zeros(num_of_zeros, device=my_device)
        to_mutate = torch.cat((ones_to_mutate, zeros_to_mutate), -1).reshape(-1)

        idxs = torch.randperm(ln, device=my_device)

        to_mutate_one_zeros = torch.gather(to_mutate, 0, idxs)

        rnd2 = torch.rand(ln, device=my_device)

        update = rnd2 * to_mutate_one_zeros + (1 - to_mutate_one_zeros) * update

        update = update.reshape(shape)
        new_weights.append((k1, update))

    return new_weights


class TransformerPool(object):

    def __init__(self, config, model_num):
        super().__init__()
        print("Creating transformers")
        self.trainers = []
        for i in range(model_num * 2):
            trainer = KarpathyRunner(config)
            self.trainers += [trainer]
            print(f"Transformer created {i}")

    def acquire(self):
        return self.trainers.pop()

    def release(self, trainer):
        self.trainers += [trainer]


class NeuralXY(XY):
    def __init__(self, name: str, data: str, params, trainer, transformer_pool):
        super().__init__(name, data)
        self.params = params
        self.trainer = trainer
        self.transformer_pool = transformer_pool

    def crossover_transformer(self, xy1, xy2):
        trainer = self.transformer_pool.acquire()

        if self.params.use_neural_crossover:
            xy1_weights = xy1.get_transformer_weights()
            xy2_weights = xy2.get_transformer_weights()

            new_weights = neural_crossover_and_mutate(xy1_weights, xy2_weights, my_device=self.params.my_device)

            trainer.set_weights(new_weights)

        return trainer

    def crossover(self, xy2: 'XY', name: str, xy_data_size: int) -> 'XY':
        xy1, xy2 = (self, xy2) if random.random() > 0.5 else (xy2, self)

        trainer = self.crossover_transformer(xy1, xy2)
        new_data = crossover_string(xy1.data, xy2.data, xy_data_size)

        return NeuralXY(name, new_data, self.params, trainer, self.transformer_pool)

    def mutate(self, mutation_p: float, xy_data_size: int) -> None:
        super().mutate(mutation_p, xy_data_size)
        # model_weights = self.get_transformer_weights()

    def get_transformer_weights(self):
        model_weights = self.trainer.get_weights()
        # model_weights = {k: v.cpu() for k, v in model_weights.items()}
        return model_weights

    def destroy(self):
        self.transformer_pool.release(self.trainer)

    @staticmethod
    def createNeuralXY(name, xy_data_size: int, params, transformer_pool):
        data = gen_rnd_chars(xy_data_size)
        trainer = transformer_pool.acquire()
        return NeuralXY(name, data, params, trainer, transformer_pool)

    def __str__(self):
        return "f={f}, d={data}".format(
            f=1/self.f,
            data="",
        )


class TargetStringTransformerEvaluator(AbstractEvaluator):
    def __init__(self, config):
        self.target = "ABABAGALAMAGAABABAGALAMAGAABABAGALAMAGAABABAG"
        self.xy_data_size_const = len(self.target)
        self.dataloader = GptNanoDataloader(config)

    def func(self, xy) -> float:
        # data = xy.data
        # diff = random.random() * 0.001
        # diff += (self.xy_data_size_const - str_diff(self.target, data))

        for i in range(5):
            x, y = self.dataloader.get_train_batch()
            logits, loss = xy.trainer.learn(x, y)

        # xy.trainer.train_iterate(100, self.dataloader.get_train_batch())

        val = xy.trainer.evaluate(self.dataloader.get_train_batch, 20).item()

        return val

    def get_xy_len(self) -> int:
        return self.xy_data_size_const

    def is_inverse_fitness(self):
        return True


class GAModelRunnner(AbstractModelRunnner):

    def __init__(self, gpu_num, model_num, params):
        self.gpu_num = gpu_num
        self.model_num = model_num
        self.params = params

        self.config = TransformerConfig(params.my_device)

        self.population_size = params.ga_population_size
        self.transformer_pool = TransformerPool(self.config, self.population_size)

        self.log_file = self.setup_logger(gpu_num, params)

        if self.params.use_neural_estimator:
            self.estimator_trainer = KarpathyRunner(self.config)

        self.training_set = set()

        def neural_xy_factory(i, xy_data_size):
            return NeuralXY.createNeuralXY(i, xy_data_size, params, self.transformer_pool)

        self.ga = GA(
            # TargetStringTransformerEvaluator(self.config),
            TargetStringEvaluator(),
            population_size=self.population_size,
            verbose=params.verbose,
            xy_factory=neural_xy_factory
        )
        self.ga.evaluate()
        self.ga.sort_population()

        self.start_time = time.time()

    def get_best_xy(self, n=1):
        return self.ga.get_best_pp(n)

    def replace_worst_xy(self, best_xy):
        self.ga.replace_worst_pp(best_xy)

    def neural_crossover(self, ga, params, trainer):
        children = []
        families = []
        p1, p2 = ga.select_random_parents(params.batch_size)
        pp1 = self.xy_to_data(p1)
        pp2 = self.xy_to_data(p2)
        children_data = trainer.act(pp1, pp2)

        for p1, p2, ch_data in zip(p1, p2, children_data):
            ch = XY('', ch_data)
            children += [ch]
            families += [(p1, p2, ch)]

        # pre_eval = pre_evaluate(children, None)

        # pre_eval, children, families = zip(pre_eval, children, families).sort()

        children = children[:ga.new_size]
        families = families[:ga.new_size]

        return children, families

    def pre_evaluate(self, children, trainer):
        ch_data = [self.xy_to_data(ch) for ch in children]
        ch_f = trainer.act(ch_data, ch_data)
        return ch_f

    def xy_to_data(self, p1):
        pp1 = [p.data for p in p1]
        return pp1

    @timeit("GAModelRunnner")
    def step(self, iteration_num, gpu_num):

        # sentimental_transformer = build_sentimental_transformer(env, params)
        # sentimental_trainer = RealtimeTrainer(sentimental_transformer, env, params)

        tm = time.time()
        ga = self.ga

        ga.print_population()

        if self.params.use_neural_crossover and ga.iteration > self.params.neural_crossover_iteration_threshold:  # random.random() > 0.5 and
            children, families = self.neural_crossover(ga, self.params, self.crossover_trainer)
        else:
            children, families = ga.crossover()

        for a, b, c in families:
            self.log(f"crossover,{iteration_num},{a.data},{b.data},{c.data}\n")

        # for xy in ga.population:
        #     print(crossover_trainer.modules['transformer'].state_dict())

        children = ga.mutate(children)

        for c in children:
            self.log(f"mutated,{iteration_num},{c.data}\n")

        for xy in ga.get_worst_pp(ga.new_size):
            xy.destroy()

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        for c in ga.population:
            self.log(f"evaluated,{iteration_num},{c.f},{c.data}\n")

        self.learn_crossover(families)

        # sentimental_trainer.learn_accumulate()

        # model_weights = self.crossover_transformer['transformer'].state_dict()
        # model_weights = {k: v.cpu() for k, v in model_weights.items()}

        ga.iteration += 1

        tm_new = time.time()

        print(f"Time of iteration is {tm_new - tm}, it={ga.iteration}, gpu={gpu_num}")

        self.log(f"iteration_time,{iteration_num},{tm_new - tm}\n")

        tm = tm_new

        # end_time = time.time()
        #
        # print(f"Total time taken = {end_time - start_time}")
        # print(f"Average time per iteration = {(end_time - start_time) / iterations}")

    def learn_crossover(self, families):
        if self.params.use_neural_crossover:
            for a, b, c in families:
                df = (c.f - max(a.f, b.f))
                # if df < 0:
                #     df = df * 0.001
                # for _ in range(params.batch_size):
                df = 1
                self.training_set.add((a.data, b.data, c.data, df))

            for (a, b, c, df) in random.sample(self.training_set, min(self.params.batch_size * 1, len(self.training_set))):
                self.crossover_trainer.learn_accumulate(a, b, c, df)

    def log(self, log_line):
        if self.log_file:
            self.log_file.write(log_line)

    def setup_logger(self, gpu_num, params):
        if params.log_ga_into_file:
            current_date_time = time.strftime("%H-%M-%S", time.localtime())
            return open(f"./logs/evolution-{gpu_num}-{current_date_time}.txt", "w")
        else:
            return None