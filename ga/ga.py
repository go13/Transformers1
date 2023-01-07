import random
import string
import numpy as np
from abc import ABC, abstractmethod

from src.utils import str_diff, words2string

data_dict = (string.ascii_uppercase + string.digits)
mutation_p_const = 0.05
new_percentage = 0.8


def gen_rnd_chars(ln):
    return words2string(random.choices(data_dict, k=ln))


def replace_char_at_index(org_str, index, replacement):
    ''' Replace character at index in string org_str with the
    given replacement character.'''
    new_str = org_str
    if index < len(org_str):
        new_str = org_str[0:index] + replacement + org_str[index + 1:]
    return new_str


def mutate(d, mutation_p, xy_data_size):
    for i in range(0, xy_data_size):
        if random.random() < mutation_p:
            v = gen_rnd_chars(1)[0]
            d = replace_char_at_index(d, i, v)
    return d


class AbstractEvaluator(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def func(self, data):
        pass

    @abstractmethod
    def get_xy_len(self):
        pass


class TargetStringEvaluator(AbstractEvaluator):

    def __init__(self):
        super().__init__()
        self.target = "ABABAGALAMAGAABABAGALAMAGAABABAGALAMAGAABABAG"
        self.xy_data_size_const = len(self.target)

    def func(self, data):
        diff = random.random() * 0.001
        # for i in range(len(xy.data)):
        #     diff += 0 if target[i] != xy.data[i] else 1
        diff += (self.xy_data_size_const - str_diff(self.target, data))
        return diff

    def get_xy_len(self):
        return self.xy_data_size_const


class XY(object):

    def __init__(self, name, data, p1=-1, p2=-1):
        self.data = data
        self.name = name
        self.p1 = p1
        self.p2 = p2
        self.f = None

    def crossover(self, xy2, name, xy_data_size):
        d1, d2 = (self.data, xy2.data) if random.random() > 0.5 else (xy2.data, self.data)

        cp = random.randint(0, xy_data_size - 1)
        # for i in range(0, xy_data_size):
        #     v = d1[i] if i < cp else d2[i]
        #     new_data += [v]

        new_data = d1[0:cp] + d2[cp: xy_data_size]

        # new_data = join_to_string(new_data)

        return XY(name, new_data, self.name, xy2.name)

    def mutate(self, mutation_p, xy_data_size):
        self.data = mutate(self.data, mutation_p, xy_data_size)

    def __str__(self):
        return "n={name}({p1}, {p2}), f={f}, d={data}".format(
            name=self.name,
            f=self.f,
            data=self.data,
            p1=self.p1,
            p2=self.p2
        )


class GA(object):

    def __init__(
            self,
            evaluator: TargetStringEvaluator,
            population_size=20,
            mutation_p=mutation_p_const,
            verbose=True
    ):
        self.verbose = verbose
        self.iteration = 0
        self.population_size = population_size
        self.evaluator = evaluator
        self.mutation_p = mutation_p
        self.mutation_enabled = True
        self.xy_data_size = evaluator.get_xy_len()
        self.population = self.generate(population_size, self.xy_data_size)
        self.new_size = int(new_percentage * self.population_size)

    @staticmethod
    def generate(population_size, xy_data_size):
        pp = []
        for i in range(population_size):
            data = gen_rnd_chars(xy_data_size)
            xy = XY(i, data)
            pp.append(xy)
        return pp

    def step(self):
        self.evaluate()
        self.sort_population()

        self.print_population()

        children = self.crossover()
        children = self.mutate(children)

        self.update_bottom(children)

        self.iteration += 1

    def add(self, xy):
        self.population.append(xy)

    def get_population(self):
        return self.population

    def get_statistics(self):
        x_values = [p.data for p in self.population]
        f_values = [p.f for p in self.population]

        # std_x_val = np.std(f_values)
        mean_f_val = np.mean(f_values)
        std_f_val = np.std(f_values)
        min_f_val = np.min(f_values)
        max_f_val = np.max(f_values)

        return mean_f_val, std_f_val, min_f_val, max_f_val

    def evaluate(self):
        for xy in self.population:
            xy.f = self.evaluator.func(xy.data)

    def mutate(self, pp, mp=None):
        # if not self.mutation_enabled:
        #     return

        mp_ = mp if mp else self.mutation_p

        for p in pp:
            p.mutate(mp_, self.xy_data_size)

        return pp

    def crossover(self):
        children = self.generate_crossover(self.new_size)

        return children

    def generate_crossover(self, new_size):
        new_population = []
        for i in range(new_size):
            xy1 = self.get_random_xy(self.population)
            xy2 = self.get_random_xy(self.population)

            child = xy1.crossover(xy2, i, self.xy_data_size)
            # child.mutate(self.mutation_p)

            new_population.append(child)

        return new_population

    def update_bottom(self, new_population):
        # new_population = new_population.copy()
        left = len(self.population) - len(new_population)
        for i in range(left):
            p = self.population[i]
            new_population.append(p)
        self.population = new_population

    def update_population(self, new_population):
        self.population = new_population

    def sort_population(self):
        self.sort_pp(self.population)

    def get_best_pp(self, pp, n):
        return pp[0: n]

    def get_worst_pp(self, n):
        return self.population[-n:]

    def sort_pp(self, pp):
        pp.sort(key=lambda xy: xy.f, reverse=True)

    def print_population(self):
        if self.verbose:
            print(f"iteration={self.iteration}")
            for xy in self.population:
                print(f"xy: {xy}")

    def get_random_n_xy(self, pp, n):
        pp = pp.copy()
        res = []
        for i in range(n):
            xy = self.get_random_xy(pp)
            pp.remove(xy)
            res += [xy]
        return res

    @staticmethod
    def get_random_xy(population):
        return GA.get_random_xy_with_position(population)[0]

    @staticmethod
    def get_random_xy_with_position(population):
        sm = 0
        for xy in population:
            sm += xy.f

        res = population[0]
        rnd = random.random() * sm
        sm = 0

        ind = 0
        for xy in population:
            if xy.f + sm >= rnd:
                res = xy
                break
            sm += xy.f
            ind += 1

        return res, ind
