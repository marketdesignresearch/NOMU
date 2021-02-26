# Libs
import numpy as np

# Internal
from bayesian_optimization.acquisition.mip_acquisition_function import MIPAcquisitionFunction

# Type hints
from typing import *
if TYPE_CHECKING:
    from configobj import ConfigObj
    from bayesian_optimization.context.context import Context


class UpperBound(MIPAcquisitionFunction):
    """Implementation of the very simple acquisition value that just adds the mean
    and the variance to result in the upper bound value
    """

    def __init__(
            self,
            factor: float = 1.0,
    ):
        """constructor

        :param factor: multiplier for the sigma
        :param scale_mean_width: scale the simga part of the acq so that the mean width is equal to the given value
        """
        super().__init__()
        self.factor = factor

    def evaluate(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float) -> np.ndarray:
        """calculate the acquisition value given a sequence of mean (mu) and variance (sigma).
        For each entry in the input array they are combined into one output value.

        :param mu: array of mean values
        :param sigma: array of variance values
        :param incumbent: previously best target-evaluation
        :return: array of acquisition value
        """
        return np.asarray([self._single_evaluate(x, y, incumbent) for x, y in zip(mu, self.factor * sigma)])

    def _single_evaluate(self, mu: np.array, sigma: np.array, incumbent: np.array) -> np.array:
        """calculate the acquisition valule for one singe mean and variance value pair

        :param mu: mean value
        :param sigma:  variance value
        :param incumbent: previously best target value
        :return: acquisition value
        """
        if sigma < 0.0:
            return mu
        else:
            return mu + (self.factor * sigma)

    def single_expression(self, model, mu: float, sigma: float, incumbent: float):
        """calculation of the acquisition function value base on a expression
        formulation applicable for a MIP

        :param model: cplex model
        :param mu: mean prediction
        :param sigma: variance of the prediction
        :param incumbent: previously best evaluation
        :return: Acquisition function formulation expression
        """
        return model.max(mu + (self.factor * sigma), mu)

    def evaluate_inspector_acq(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> np.array:
        """secondary interface to evaluate the acquisition function post experiment from the context

        :param mu: mean value
        :param sigma: variance value
        :param context: Context of the run
        :param step: bayesian optimization step
        :return: acq values
        """
        return self.evaluate(mu, sigma, 0.0)

    def evaluate_inspector_mu_sig(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> Tuple[np.array, np.array]:
        """secondary interface to evaluate the mean and uncertainty post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: mu and sigma estimates
         """
        return mu, sigma

    @classmethod
    def read_from_config(cls, config: 'ConfigObj') -> 'UpperBound':
        """reads the configuration from the config file and creates the mean width scaled upper bound
        acquisition function accordingly.
        :param config: config parser instance
        :return: acquisitions function instance
        """
        return cls(
            factor=config.as_float("factor"),
        )