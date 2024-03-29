from t3_karpathy.enhanced_karpathy_transformer import EnhancedKarpathyRunner


class TransformerPool(object):

    def __init__(self, params, model_num, runner_factory):
        super().__init__()
        print("Creating transformers")
        self.params = params
        self.trainers = []
        for i in range(model_num * 2):
            trainer = runner_factory()
            self.trainers += [trainer]
            print(f"Transformer created {i}")
            pass

    def acquire(self):
        if self.params.use_transformer_transformer:
            return self.trainers.pop()
        else:
            return None

    def release(self, trainer):
        if self.params.use_transformer_transformer:
            self.trainers += [trainer]
        else:
            pass