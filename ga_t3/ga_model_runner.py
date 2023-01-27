import time
import random

import numpy as np
import torch
from ga.ga import GA, XY, gen_rnd_chars, crossover_string, AbstractEvaluator, TargetStringEvaluator, get_random_xy, sanitize
from t3_karpathy.autoencoder_transformer import AutoencoderAccumulativeTrainer
from t3_karpathy.crossover_transformer import CrossoverAccumulativeTrainer
from t3_karpathy.sentimental_transformer import SentimentalAccumulativeTrainer
from ga_t3.base_model_runner import AbstractModelRunnner
from ga_t3.transformer_pool import TransformerPool
from src.performance_utils import timeit
from t3_karpathy.gpt_nano_dataloader import GptNanoDataloader
from t3_karpathy.transformer_config import TransformerConfig


def transformer_neural_crossover_and_mutate(xy1_weights, xy2_weights, my_device):
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


class NeuralXY(XY):
    def __init__(self, data: str, params):
        super().__init__(data)
        self.params = params

    def crossover(self, xy2: 'XY', xy_data_size: int) -> 'XY':
        xy1, xy2 = (self, xy2) if random.random() > 0.5 else (xy2, self)

        new_data = crossover_string(xy1.data, xy2.data, xy_data_size)

        return NeuralXY(new_data, self.params)

    def mutate(self, mutation_p: float, xy_data_size: int, vocab) -> None:
        super().mutate(mutation_p, xy_data_size, vocab)

    def get_transformer_weights(self):
        pass

    def destroy(self):
        pass

    @staticmethod
    def generate_new_neural_xy(xy_data_size: int, params):
        data = gen_rnd_chars(xy_data_size)
        return NeuralXY(data, params)

    def __str__(self):
        return "id={id}, f={f}, d={data}".format(
            id=self.id,
            f=self.f,
            data=sanitize(self.data),
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


class GAModelRunner(AbstractModelRunnner):

    def __init__(self, gpu_num, model_num, params):
        self.gpu_num = gpu_num
        self.model_num = model_num
        self.params = params

        self.config = TransformerConfig(params.my_device)
        self.config.block_size = 45

        self.population_size = params.ga_population_size
        self.transformer_pool = TransformerPool(self.config, params, self.population_size)

        self.log_file = self.setup_logger(gpu_num, params)

        # if self.params.use_neural_estimator:
        self.config.vocab_size = self.config.token_codec.vocab_size
        self.vocab = self.config.token_codec.vocab

        if self.params.use_neural_estimator:
            self.accumulative_runner = SentimentalAccumulativeTrainer(self.config)

        if self.params.use_neural_crossover:
            self.crossover_trainer = CrossoverAccumulativeTrainer(self.config)

        if self.params.use_neural_autoencoder:
            self.autoencoder = AutoencoderAccumulativeTrainer(self.config)

        def neural_xy_factory(xy_data_size):
            return NeuralXY.generate_new_neural_xy(xy_data_size, params)

        self.ga = GA(
            TargetStringEvaluator(),
            population_size=self.population_size,
            verbose=params.verbose,
            mutation_p=params.ga_mutation_p,
            xy_factory=neural_xy_factory,
            vocab=self.config.token_codec.vocab
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

        prediction = self.crossover_trainer.predict_list(pp1, pp2)

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

        ga.sort_population()
        ga.print_population()

        children, families = self.process_children(ga)

        # for a, b, c in families:
        #     self.log(f"crossover,{iteration_num},{a.data},{b.data},{c.data}\n")

        # for xy in ga.population:
        #     print(crossover_trainer.modules['transformer'].state_dict())

        for c in children:
            self.log(f"mutated,{iteration_num},{c.data}\n")

        if self.params.ga_generate_only_unique_xy:
            self.learn_neural_estimator(ga.population)
        else:
            self.learn_neural_estimator(ga.population)

        for xy in ga.get_worst_pp(ga.new_size):
            xy.destroy()

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        for c in ga.population:
            self.log(f"evaluated,{iteration_num},{c.f},{sanitize(c.data)}\n")

        self.learn_crossover(families)
        self.learn_neural_autoencoder(ga.population)

        ga.iteration += 1

        tm_new = time.time()

        print(f"Time of iteration is {tm_new - tm}, it={ga.iteration}, gpu={gpu_num}")

        self.log(f"iteration_time,{iteration_num},{tm_new - tm}\n")

        tm = tm_new

        # end_time = time.time()
        #
        # print(f"Total time taken = {end_time - start_time}")
        # print(f"Average time per iteration = {(end_time - start_time) / iterations}")

    def process_children(self, ga):
        mp = ga.mutation_p
        generated_children = []
        generated_families = []
        just_created_children_dict = dict()
        i = 0
        while i < 4:
            if self.params.use_neural_estimator and ga.iteration > self.params.neural_estimator_iteration_start:
                children, families = self.generate_crossover(ga.new_size * 10)
                children = ga.mutate(children, mp)
                data_list = [xy.data for xy in children]
                estimations_list = self.accumulative_runner.predict_list(data_list)
                estimated_children_families = list(zip(children, estimations_list, families))
                sorted_children_families = sorted(estimated_children_families, key=lambda x: x[1], reverse=True)
                children = [x[0] for x in sorted_children_families]
                families = [x[2] for x in sorted_children_families]
            else:
                children, families = self.generate_crossover(ga.new_size)
                children = ga.mutate(children)

            if self.params.ga_generate_only_unique_xy:
                for c, f in list(zip(children, families)):
                    c_data = c.data
                    if c_data not in self.vocab and c_data not in just_created_children_dict:
                        generated_children += [c]
                        just_created_children_dict[c_data] = c_data
                        generated_families += [f]
                    else:
                        print(f"Duplicate child {c_data}")

            else:
                generated_children += children
                generated_families += families

            i += 1
            if len(generated_children) < ga.new_size:
                mp *= 2
                print(f"Mutation probability increased to {mp}, generated unique {len(generated_children)} children")
            else:
                break

        children = generated_children[0:ga.new_size]
        families = generated_families[0:ga.new_size]
        return children, families

    def generate_crossover(self, new_size):
        if (self.params.use_neural_crossover or self.params.use_neural_autoencoder) and \
                (self.ga.iteration > self.params.neural_crossover_iteration_start or self.ga.iteration > self.params.use_neural_autoencoder_iteration_start):

            if random.random() < self.params.neural_crossover_regular_crossover_prob:
                return self.ga.generate_crossover(new_size)

            xy1_list = []
            xy2_list = []

            xy1_data_list = []
            xy2_data_list = []

            new_families = []
            new_population = []
            for i in range(new_size):
                xy1 = get_random_xy(self.ga.population)
                xy2 = get_random_xy(self.ga.population)

                xy1_list += [xy1]
                xy2_list += [xy2]

                xy1_data_list += [xy1.data]
                xy2_data_list += [xy2.data]

            if self.params.use_neural_crossover:
                predicted_list = self.crossover_trainer.predict_list(xy1_data_list, xy2_data_list)

            if self.params.use_neural_autoencoder:
                predicted_list = self.autoencoder.predict_list(xy1_data_list, xy2_data_list)

            for i in range(new_size):
                new_data = predicted_list[i]
                xy1 = xy1_list[i]
                xy2 = xy2_list[i]

                child = NeuralXY(new_data, self.params)

                family = (xy1, xy2, child)

                new_population += [child]
                new_families += [family]

            return new_population, new_families
        else:
            return self.ga.generate_crossover(new_size)

    def learn_crossover(self, families):
        def f_transform(z):
            if z > 0:
                return z
            return 1 / (1 + np.exp(-z - 4))

        if self.params.use_neural_crossover:
            for x1, x2, y in families:
                f = y.f# - max(x1.f, x2.f)
                if f <= 0:
                    continue

                f = f_transform(f)
                self.crossover_trainer.add_sample(x1.data, x2.data, y.data, f)

            av_loss, total_samples = self.crossover_trainer.train(n = self.params.neural_crossover_iterations_per_ga_iteration)
            print(f"Crossover average loss={av_loss}, total_samples={total_samples}")

    def learn_neural_autoencoder(self, population):
        if self.params.use_neural_autoencoder:
            for xy in population:
                self.autoencoder.add_sample(xy.data, xy.data)

            av_loss, total_samples= self.autoencoder.train(1)
            print(f"Autoencoder average loss={av_loss}, total_samples={total_samples}")

    def learn_neural_estimator(self, new_samples):
        if self.params.use_neural_estimator:
            for xy in new_samples:
                self.accumulative_runner.add_sample(xy.data, xy.f)

            av_loss, total_samples = self.accumulative_runner.train(n=self.params.ga_neural_estimator_iterations_per_ga_iteration)
            print(f"Estimator average loss={av_loss}, total_samples={total_samples}")

    def log(self, log_line):
        if self.log_file:
            self.log_file.write(log_line)

    def setup_logger(self, gpu_num, params):
        if params.log_ga_into_file:
            current_date_time = time.strftime("%H-%M-%S", time.localtime())
            return open(f"./logs/evolution-{gpu_num}-{current_date_time}.txt", "w")
        else:
            return None
