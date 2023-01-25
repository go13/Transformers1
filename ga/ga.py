import random
import string
import numpy as np
from abc import ABC, abstractmethod
from src.utils import str_diff, words2string

data_dict = string.ascii_uppercase + string.digits
mutation_p_const = 0.05
new_percentage = 0.7


def gen_rnd_chars(ln: int) -> str:
    return words2string(random.choices(data_dict, k=ln))


def replace_char_at_index(org_str, index, replacement):
    ''' Replace character at index in string org_str with the
    given replacement character.'''
    new_str = org_str
    if index < len(org_str):
        new_str = org_str[0:index] + replacement + org_str[index + 1:]
    return new_str


def mutate_string(d: str, mutation_p: float, xy_data_size: int) -> str:
    for i in range(0, xy_data_size):
        if random.random() < mutation_p:
            v = gen_rnd_chars(1)[0]
            d = replace_char_at_index(d, i, v)
    return d


def crossover_string(d1, d2, xy_data_size):
    cp = random.randint(0, xy_data_size - 1)
    new_data = d1[0:cp] + d2[cp: xy_data_size]
    return new_data


def sort_pp(pp):
    pp.sort(key=lambda xy: xy.f, reverse=True)
    return pp


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


def get_multi_random_xy(population, n=1):
    return [get_random_xy(population) for i in range(n)]


def get_random_xy(population):
    return get_random_xy_with_position(population)[0]


class AbstractEvaluator(ABC):
    @abstractmethod
    def func(self, data: 'XY') -> float:
        pass

    @abstractmethod
    def get_xy_len(self) -> int:
        pass

    def is_inverse_fitness(self):
        return False


class TargetStringEvaluator(AbstractEvaluator):
    def __init__(self):
        self.target = "ABABAGALAMAGAABABAGALAMAGAABABAGALAMAGAABABAG"
        self.xy_data_size_const = len(self.target)

    def func(self, xy) -> float:
        data = xy.data
        diff = random.random() * 0.001
        diff += (self.xy_data_size_const - str_diff(self.target, data))
        return diff

    def get_xy_len(self) -> int:
        return self.xy_data_size_const

    def is_inverse_fitness(self):
        return False


class XY(object):

    def __init__(self, name: str, data: str, p1: str = "a", p2: str = "e"):
        self.data = data
        self.name = name
        self.p1 = p1
        self.p2 = p2
        self.f = None

    def crossover(self, xy2: 'XY', name: str, xy_data_size: int) -> 'XY':
        d1, d2 = (self.data, xy2.data) if random.random() > 0.5 else (xy2.data, self.data)

        new_data = crossover_string(d1, d2, xy_data_size)

        return XY(name, new_data, self.name, xy2.name)

    def mutate(self, mutation_p: float, xy_data_size: int) -> None:
        self.data = mutate_string(self.data, mutation_p, xy_data_size)

    def reuse(self, name: str, data: str, p1: str, p2: str) -> 'XY':
        self.data = data
        self.name = name
        self.p1 = p1
        self.p2 = p2
        self.f = None
        return self

    def __str__(self):
        return "n={name}({p1}, {p2}), f={f}, d={data}".format(
            name=self.name,
            f=self.f,
            data=self.data,
            p1=self.p1,
            p2=self.p2
        )

    @staticmethod
    def createXY(name, xy_data_size: int):
        data = gen_rnd_chars(xy_data_size)
        return XY(name, data)


class GA(object):
    def __init__(
            self,
            evaluator: AbstractEvaluator,
            population_size=20,
            mutation_p=mutation_p_const,
            xy_factory=XY.createXY,
            verbose=True,
            inverse_fitness=False
    ):
        self.iteration = 0
        self.inverse_fitness = evaluator.is_inverse_fitness()
        self.xy_factory = xy_factory
        self.verbose = verbose
        self.population_size = population_size
        self.evaluator = evaluator
        self.mutation_p = mutation_p
        self.mutation_enabled = True
        self.xy_data_size = evaluator.get_xy_len()
        self.population = self.generate(population_size, self.xy_data_size)
        self.new_size = int(new_percentage * self.population_size)
        self.transformer_pool = None

    def generate(self, population_size, xy_data_size):
        pp = []
        for i in range(population_size):
            xy = self.xy_factory(i, self.xy_data_size)
            pp.append(xy)
        return pp

    def step(self):
        self.evaluate()
        self.sort_population()

        self.print_population()

        children, families = self.crossover()
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
            xy.f = self.evaluator.func(xy)

        self.inverse_f()

    def inverse_f(self):
        if self.inverse_fitness:
            for xy in self.population:
                xy.f = 1 / xy.f

            # mx = self.population[0].f
            # for xy in self.population:
            #     if xy.f > mx:
            #         mx = xy.f

            # for xy in self.population:
            #     xy.f = mx - xy.f

    def mutate(self, pp, mp=None):
        # if not self.mutation_enabled:
        #     return

        mp_ = mp if mp else self.mutation_p

        for p in pp:
            p.mutate(mp_, self.xy_data_size)

        return pp

    def crossover(self):
        return self.generate_crossover(self.new_size)

    def select_random_parents(self, new_size):
        def rxy():
            return get_random_xy(self.population)
        return [rxy() for _ in range(new_size)], [rxy() for _ in range(new_size)]

    def generate_crossover(self, new_size):
        new_families = []
        new_population = []
        for i in range(new_size):
            xy1 = get_random_xy(self.population)
            xy2 = get_random_xy(self.population)

            child = xy1.crossover(xy2, '', self.xy_data_size)

            family = (xy1, xy2, child)

            new_population += [child]
            new_families += [family]

        return new_population, new_families

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
        self.population = sort_pp(self.population)

    def get_best_pp(self, n):
        return self.population[: n]

    def get_worst_pp(self, n):
        return self.population[-n:]

    def replace_worst_pp(self, new_pp):
        for i in range(len(new_pp)):
            self.population[-i - 1] = new_pp[i]

    def print_population(self):
        if self.verbose or self.iteration % 100 == 0:
            print(f"iteration={self.iteration}")
            for xy in self.population:
                print(f"xy: {xy}")

    def get_random_n_xy(self, pp, n):
        pp = pp.copy()
        res = []
        for i in range(n):
            xy = get_random_xy(pp)
            pp.remove(xy)
            res += [xy]
        return res
