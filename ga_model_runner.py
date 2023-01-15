import time
import random

from envs import build_env
from ga.ga import GA, TargetStringEvaluator, XY
from base_model_runners import AbstractModelRunnner
from src.performance_utils import timeit
from t2.realtime_trainer import RealtimeTrainer
from t2.transformer import build_transformer


class GAModelRunnner(AbstractModelRunnner):

    def __init__(self, gpu_num, model_num, params, env):
        self.gpu_num = gpu_num
        self.model_num = model_num
        self.params = params

        current_date_time = time.strftime("%H-%M-%S", time.localtime())
        self.log_file = open(f"evolution-{gpu_num}-{current_date_time}.txt", "w")

        self.crossover_transformer = build_transformer(env, params)
        self.crossover_trainer = RealtimeTrainer(self.crossover_transformer, env, params)

        self.training_set = set()

        self.ga = GA(TargetStringEvaluator())
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

    @classmethod
    def create(self, **kwargs):
        return GAModelRunnner(**kwargs)

    @timeit("GAModelRunnner")
    def step(self, iteration_num, gpu_num):

        # sentimental_transformer = build_sentimental_transformer(env, params)
        # sentimental_trainer = RealtimeTrainer(sentimental_transformer, env, params)

        tm = time.time()
        ga = self.ga

        ga.print_population()

        if ga.iteration > 200 and False:  # random.random() > 0.5 and
            children, families = self.neural_crossover(ga, self.params, self.crossover_trainer)
        else:
            children, families = ga.crossover()

        for a, b, c in families:
            self.log_file.write(f"crossover,{iteration_num},{a.data},{b.data},{c.data}\n")

        # for xy in ga.population:
        #     print(crossover_trainer.modules['transformer'].state_dict())

        children = ga.mutate(children)

        for c in children:
            self.log_file.write(f"mutated,{iteration_num},{c.data}\n")

        ga.update_bottom(children)

        ga.evaluate()
        ga.sort_population()

        for c in ga.population:
            self.log_file.write(f"evaluated,{iteration_num},{c.f},{c.data}\n")

        # learn crossover result
        # for a, b, c in families:
        #     df = (c.f - max(a.f, b.f))
        #     # if df < 0:
        #     #     df = df * 0.001
        #     # for _ in range(params.batch_size):
        #     df = 1
        #     self.training_set.add((a.data, b.data, c.data, df))
        #
        # for (a, b, c, df) in random.sample(self.training_set, min(self.params.batch_size * 1, len(self.training_set))):
        #     self.crossover_trainer.learn_accumulate(a, b, c, df)

        # sentimental_trainer.learn_accumulate()

        # model_weights = self.crossover_transformer['transformer'].state_dict()
        # model_weights = {k: v.cpu() for k, v in model_weights.items()}

        ga.iteration += 1

        tm_new = time.time()

        print(f"Time of iteration is {tm_new - tm}, it={ga.iteration}, gpu={gpu_num}")

        self.log_file.write(f"iteration_time,{iteration_num},{tm_new - tm}\n")

        tm = tm_new

        # end_time = time.time()
        #
        # print(f"Total time taken = {end_time - start_time}")
        # print(f"Average time per iteration = {(end_time - start_time) / iterations}")