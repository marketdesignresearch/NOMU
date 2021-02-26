# Libs
import numpy as np

# Internal
from bayesian_optimization.acquisition.decorators.abstract_decorator import AbstractDecorator
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction

# Type hinting
from typing import *
if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from configobj import ConfigObj


class BoundedR(AbstractDecorator):
    """Decorator that bounds the uncertainty before applied incorporated into the acquisition function
    """

    def __init__(self, acq_to_decorate, r_max: float = 2.0, r_min: float = 1e-6):
        super().__init__(acq_to_decorate)
        self.r_max = r_max
        self.r_min = r_min

    @staticmethod
    def relu(x: np.array) -> np.array:
        """ReLU activation wrapper
        :param x: input x
        :return: ReLU result
        """
        return np.maximum(0, x)

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
        sigma = self.r_max*(1.-np.exp(-(self.relu(sigma)+self.r_min)/self.r_max))
        return self.acq_to_decorate.evaluate(mu, sigma, incumbent)

    def evaluate_inspector_mu_sig(
            self, mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> Tuple[np.array, np.array]:
        """secondary interface to evaluate the uncertainty post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: mu and sigma estimates
         """
        sigma = self.r_max*(1.-np.exp(-(self.relu(sigma)+self.r_min)/self.r_max))
        return self.acq_to_decorate.evaluate_inspector_mu_sig(mu, sigma, context, step)

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

    @staticmethod
    def read_from_config(
            config: 'ConfigObj',
            context: 'Context',
            base_acq: 'AcquisitionFunction'
    ) -> 'AcquisitionFunction':
        """reads the configuration from the config file and add the decorator configured accordingly to the give acq-function
        :param config: config parser instance
        :param base_acq: Acquisition function wo wrap
        :return: acquisitions function instance
        """
        return BoundedR(
            acq_to_decorate=base_acq,
            r_max=config.as_float("r_max"),
            r_min=config.as_float("r_min"),
        )
