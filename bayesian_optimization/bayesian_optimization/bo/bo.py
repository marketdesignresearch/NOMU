# Libs
import random
import numpy as np
import os
from tensorflow.keras import backend as K
# ------------------------------------------------------------------------- #
# disable eager execution for tf.__version__ 2.3
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------- #
# Internals
from bayesian_optimization.context.context import Context

# Type hinting
from typing import *
if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context


class BO:
    """Class implementing the Bayesian optimization algorithm
    """

    INSP_SAMPLES_X = "samples_x"
    INSP_SAMPLES_Y = "samples_y"
    INSP_NEW_SAMPLE_X = "new_sample_x"
    INSP_NEW_SAMPLE_Y = "new_sample_y"

    """Bayesian optimization algorithm implementation
    """
    def __init__(
            self,
            context: 'Context',
            maxiter: int,
            acq_threshold: float = None
    ):
        """constructor

        :param context: context defining all sub processes of the bayesian optimization
        :param maxiter: maximum number of iterations for the bayesian optimization
        :param acq_threshold:
        """
        self.context = context
        self.optimizer = self.context.acq_optimizer
        self.maxiter = maxiter
        self.acq_threshold = acq_threshold

    def run(self, callback_method: Callable, save_to_files=True, seed=0):
        """execution of the Bayesian optimization algorithm.
        First calls the acquisition function optimizers to return the inputvalue
        which optimizes (maximizes) the acquisition function.
        Then it evaluates the target function at this point and adds this evaluations as a new sample
        and repeats the process

        :param callback_method: method that evaluates a input values and returns an appropriate target value
        :return:optimal y found, input for the found optima
        """
        temp_acq = 0
        for i in range(0, self.maxiter):

            self.context.set_bo_step(i)
            argmax, maxvalmu, maxvalsig, max_acq = self.optimizer.get_optima(np.copy(self.context.samples_x), np.copy(self.context.samples_y))
            new_sample_x = [argmax]
            new_sample_y = callback_method([argmax])

            if self.acq_threshold and abs(max_acq-temp_acq) < self.acq_threshold:
                print("Early stopping after {} Iterations, ACQ-change ({}) < {}".format(i, abs(max_acq-temp_acq), self.acq_threshold))
                break
            temp_acq = max_acq
            self.context.inspector.dump_data(self.INSP_SAMPLES_X, self.context.samples_x)
            self.context.inspector.dump_data(self.INSP_SAMPLES_Y, self.context.samples_y)
            self.context.inspector.dump_data(self.INSP_NEW_SAMPLE_X, new_sample_x)
            self.context.inspector.dump_data(self.INSP_NEW_SAMPLE_Y, new_sample_y)
            self.context.samples_x = np.append(self.context.samples_x, new_sample_x, axis=0)
            self.context.samples_y = np.append(self.context.samples_y, new_sample_y, axis=0)

            if save_to_files:
                path = "{}/{}".format(
                    self.context.inspector.inspector_path,
                    "BO_steps")
                os.makedirs(path, exist_ok=True)
                self.context.inspector.save_dump_as_pickle(path, self.context.bo_step)
                self.context.inspector.reset_dump()
            print("finished step {}...".format(i+1))

        return max(self.context.samples_y), self.context.samples_x[np.argmax(self.context.samples_y)]
