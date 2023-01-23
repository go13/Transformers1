import time
from ga.ga import sort_pp, get_multi_random_xy


class AbstractModelRunnner(object):

    def __init__(self, gpu_num, models_per_gpu, params):
        self.gpu_num = gpu_num
        self.models_per_gpu = models_per_gpu
        self.params = params

    def step(self, iteration_num, gpu_num, params):
        raise NotImplementedError()


class GpuRunnner(object):
    def __init__(self, gpu_num, params, model_runner_factory):
        self.inverse_fitness = False
        self.gpu_num = gpu_num
        self.models_per_gpu = params.models_per_gpu
        self.params = params
        params.my_device = 'cuda:' + str(gpu_num)
        self.runners = [model_runner_factory(self.gpu_num, i, params) for i in range(self.models_per_gpu)]
        self.select_best_of_group = params.select_best_of_group
        self.distribute_best = params.distribute_best

    def step(self, iteration_num):
        for r in self.runners:
            r.step(iteration_num, self.gpu_num)

    def iterate(self, number_of_iterations):
        for iteration_num in range(number_of_iterations):
            print(f"Starting iteration {iteration_num} on gpu {self.gpu_num}")
            start = time.time()

            self.step(iteration_num)

            end = time.time()
            print(f"Ended iteration {iteration_num} on gpu {self.gpu_num}, taken = {end - start}, time/iteration = {(end - start) / self.models_per_gpu}, models_per_gpu={self.models_per_gpu}")

            self.exchange_best_models(iteration_num)

    def exchange_best_models(self, iteration_num):
        if not self.params.exchange_best_between_gpus or not iteration_num % self.params.exchange_best_every_n_iterations == 0:
            return

        best_xy = self.get_best_xy()

        best_xy = sort_pp(best_xy)
        if self.params.ga_use_random_exchange:
            best_xy = get_multi_random_xy(best_xy, self.distribute_best)
        else:
            best_xy = best_xy[:self.distribute_best]

        self.replace_worst_xy(best_xy)

        print(f"Exchanged best models on gpu {self.gpu_num}")

    def replace_worst_xy(self, best_xy):
        for r in self.runners:
            r.replace_worst_xy(best_xy)

    def get_best_xy(self):
        top_xy = []
        for r in self.runners:
            top_xy.extend(r.get_best_xy(self.select_best_of_group))
        return top_xy