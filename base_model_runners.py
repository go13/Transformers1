class AbstractModelRunnner(object):

    def __init__(self, gpu_num, models_per_gpu, params):
        self.gpu_num = gpu_num
        self.models_per_gpu = models_per_gpu
        self.params = params

    def step(self, iteration_num, gpu_num, params):
        raise NotImplementedError()

    @classmethod
    def create(self, **kwargs):
        raise NotImplementedError()


class GpuRunnner(object):
    def __init__(self, gpu_num, models_per_gpu, params, model_runner_factory):
        self.gpu_num = gpu_num
        self.models_per_gpu = models_per_gpu
        self.params = params
        params.my_device = 'cuda:' + str(gpu_num)
        self.runners = [model_runner_factory(gpu_num=self.gpu_num, model_num=i, params=self.params) for i in range(self.models_per_gpu)]

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