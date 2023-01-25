class TransformerPool(object):

    def __init__(self, config, params, model_num):
        super().__init__()
        print("Creating transformers")
        self.params = params
        self.trainers = []
        for i in range(model_num * 2):
            # trainer = KarpathyRunner(config)
            # self.trainers += [trainer]
            # print(f"Transformer created {i}")
            pass

    def acquire(self):
        if self.params.use_evolve_transformer:
            return self.trainers.pop()
        else:
            return None

    def release(self, trainer):
        if self.params.use_evolve_transformer:
            self.trainers += [trainer]
        else:
            pass