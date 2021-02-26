# Libs
import numpy as np

# Internal
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from configobj import ConfigObj


class AbstractDecorator(AcquisitionFunction):
    """Abstract Decorator for acquisition functions
    """

    def __init__(self, acq_to_decorate: 'AcquisitionFunction') -> NoReturn:
        super().__init__()
        self.acq_to_decorate = acq_to_decorate

    def evaluate(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            incumbent: float
    ) -> np.ndarray:
        """calculate the acquisition value given a sequence of mean (mu) and variance (sigma).
        For each entry in the input array they are combined into one output value.

        :param mu: array of mean values
        :param sigma: array of variance values
        :param incumbent: previously best target-evaluation
        :return: array of acquisition value
        """
        return self.acq_to_decorate.evaluate(mu, sigma, incumbent)

    def evaluate_inspector_mu_sig(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> Tuple[np.array, np.array]:
        pass

    def single_expression(self, model,  mu: float, sigma: float, incumbent: float):
        """Expression Formulation of the acquisition function at one single point

        :param model: cplex model
        :param mu: mean prediction
        :param sigma: variance of the prediction
        :param incumbent: previously best evaluation
        :return: Acquisition function formulation expression
        """
        return self.acq_to_decorate.single_expression(model, mu, sigma, incumbent)
