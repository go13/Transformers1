from t3_karpathy.karpathy_transformer import AbstractRunner
from t3_karpathy.transformer_config import TransformerConfig


class AbstractAccumulativeTrainer(object):
    def __init__(self, config: TransformerConfig):
        self.config = config
        self.loss_hist = []

    def get_loss_history(self):
        return self.loss_hist


class StringAccumulativeTrainer(AbstractAccumulativeTrainer):
    def __init__(self, config: TransformerConfig, runner: AbstractRunner):
        super().__init__(config)
        self.runner = runner
