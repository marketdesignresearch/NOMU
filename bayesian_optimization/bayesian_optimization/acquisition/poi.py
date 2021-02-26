# Libs
import numpy as np

# Internal
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction

# Type hinting
from typing import *
if TYPE_CHECKING:
    from configparser import ConfigObj
    from bayesian_optimization.context.context import Context


class ProbabilityOfImprovement(AcquisitionFunction):
    """Implementation of the Probability of Improvement Acquisition function.
    """

    def __init__(self, xi: float = 0.0):
        """constructor

        :param xi: exploration vs. exploitation parameter. higher xi means more exploration
        """
        super().__init__()
        self.xi = xi

    def evaluate(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float) -> np.ndarray:
        """calculate the acquisition value given a sequence of mean (mu) and variance (sigma).
        For each entry in the input array they are combined into one output value.

        :param mu: array of mean values
        :param sigma: array of variance values
        :param incumbent: previously best target-evaluation
        :return: array of acquisition value
        """
        return np.asarray([self._single_evaluate(x, y, incumbent) for x, y in zip(mu, sigma)])

    def _single_evaluate(self, mu: np.array, sigma: np.array, incumbent: np.array) -> np.array:
        """calculate the acquisition valule for one singe mean and variance value pair

        :param mu: mean value
        :param sigma:  variance value
        :param incumbent: previously best target value
        :return: acquisition value
        """
        if sigma <= 0.0:
            return 0.
        else:
            return 1 * self._cdf((mu - incumbent - self.xi)/sigma)

    def evaluate_inspector_acq(self, mu: np.ndarray, sigma: np.ndarray, context: 'Context', step: int) -> np.ndarray:
        """secondary interface to evaluate the acquisition function post experiment from the context

        :param mu: mean value
        :param sigma: variance value
        :param context: Context of the run
        :param step: bayesian optimization step
        :return: acq values
        """
        return self.evaluate(mu, sigma, 0.0)

    @classmethod
    def read_from_config(cls, config: 'ConfigObj') -> 'ProbabilityOfImprovement':
        """reads the configuration from the config file and creates the EI acquisition function accordingly.
        :param config: config parser instance
        :return: acquisitions function instance
        """
        return cls(
            xi=config.as_float("xi"),
        )
