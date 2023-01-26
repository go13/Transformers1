from t3_karpathy.transformer_config import TransformerConfig


class AbstractAccumulativeTrainer(object):
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.loss_hist = []

    def get_loss_history(self):
        return self.loss_hist
