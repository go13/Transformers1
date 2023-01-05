import os
import sys
from logging import getLogger
import torch

from t2.abstract_trainer import AbstractTrainer

if torch.cuda.is_available():
    import apex

logger = getLogger()


class DataloaderTrainer(AbstractTrainer):

    def __init__(self, modules, env, params):
        super().__init__(modules, env, params)

        self.data_path = None
        # reload exported data
        if params.reload_data != '':
            assert params.export_data is False
            s = [x.split(',') for x in params.reload_data.split(';') if len(x) > 0]
            assert len(s) >= 1 and all(len(x) == 4 for x in s) and len(s) == len(set([x[0] for x in s]))

            self.data_path = {task: (train_path, valid_path, test_path) for task, train_path, valid_path, test_path in s}

            assert all(all(os.path.isfile(path) for path in paths) for paths in self.data_path.values())

            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)

        self.dataloader = {
            task: iter(self.env.create_train_iterator(task, params, self.data_path))
            for task in params.tasks
        }

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch = next(self.dataloader[task])
        except Exception as e:
            logger.error("An unknown exception of type {0} occurred in line {1} when fetching batch. "
                         "Arguments:{2!r}. Restarting ...".format(type(e).__name__, sys.exc_info()[-1].tb_lineno,
                                                                  e.args))
            if self.params.is_slurm_job:
                if int(os.environ['SLURM_PROCID']) == 0:
                    logger.warning("Requeuing job " + os.environ['SLURM_JOB_ID'])
                    os.system('scontrol requeue ' + os.environ['SLURM_JOB_ID'])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    #TODO: - still to complete
    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params

        # batch
        (x1, len1), (x2, len2), _ = self.get_batch(task)

        loss = self._learn(x1, len1, x2, len2)

        self.stats[task].append(loss.item())

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats['processed_e'] += len1.size(0)
        self.stats['processed_w'] += (len1 + len2 - 2).sum().item()