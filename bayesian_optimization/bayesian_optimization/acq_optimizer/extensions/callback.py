# Libs
import numpy as np

# Type hints
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context


class OptimizerCallback:
    """Abstract Class defining a callback which should be executed after the initial optimizing.
    This can mainly be used to further improve the optimization. One example would be to adjust the
    acquisition function and optimize again if the initially found optima is too close to an already evaluated
    point. But there are also other applications for a Optimizer Callback.

    """

    def __init__(self):
        """constructor
        """
        self.context = None
        self.run_inspections = []

    def set_context(self, context: 'Context') -> NoReturn:
        """set the context to the Callback
        :param context: context to be assigned to the callback
        :return:
        """
        self.context = context

    def run(
            self,
            new_sample_x: np.array,
            new_sample_mu: float,
            new_sample_sigma: float,
            new_sample_acq: float,
            mu: np.array,
            sigma: np.array,
            acq_values: np.array,
            iteration: int
    ) -> Tuple[np.array, float, float, float, np.array, np.array, np.array]:
        """actual callback function to be executed.

        :param new_sample_x: found optima which would be next sample in the optimizer
        :param new_sample_mu: mu value estimated for the found optima/ next sample
        :param new_sample_sigma: sigma value estimated for the found optima/ next sample
        :param new_sample_acq: acquisition value estimated for the found optima/ next sample
        :param mu: array of all mu evaluation needed for the optimization
        :param sigma: array of all sigma evaluation needed for the optimization
        :param acq_values: array of all acq evaluation needed for the optimization
        :param iteration: iteration number, how often the callback has been called (for recursive schemes)
        :return:
        """
        pass

