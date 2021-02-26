# Libs
import numpy as np
from docplex.mp.basic import Expr

# Internal
from bayesian_optimization.acquisition.mip_acquisition_function import MIPAcquisitionFunction

# Type hinting
from typing import *
if TYPE_CHECKING:
    from configobj import ConfigObj
    from bayesian_optimization.context.context import Context



class UncertaintyOnly(MIPAcquisitionFunction):
    """Implementation of the very simple acquisition value that just returns the value of the prediction
    of the uncertainty/ residual
    """

    def __init__(self):
        """constructor
        """
        super().__init__()

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
        """calculate the acquisition value for one singe mean and variance value pair

        :param mu: mean value
        :param sigma:  variance value
        :param incumbent: previously best target value
        :return: acquisition value
        """
        if sigma < 0.0:
            return 0.
        else:
            return sigma

    def single_expression(self, model, mu: float, sigma: float, incumbent: float) -> Expr:
        """calculation of the acquisition function value base on a expression
        formulation applicable for a MIP

        :param model: cplex model
        :param mu: mean prediction
        :param sigma: variance of the prediction
        :param incumbent: previously best evaluation
        :return: Acquisition function formulation expression
        """
        return model.max(sigma, 0.0)

    def evaluate_inspector_acq(self, mu: np.ndarray, sigma: np.ndarray, context: 'Context', step: int) -> np.ndarray:
        """secondary interface to evaluate the acquisition function post experiment from the context

        :param mu: mean value
        :param sigma: variance value
        :param context: Context of the run
        :param step: bayesian optimization step
        :return: acq values
        """
        return self.evaluate(mu, sigma, 0.0)
