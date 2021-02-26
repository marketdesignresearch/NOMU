# Libs
import numpy as np
from scipy.stats import norm

# Type hints
from typing import *
from typing import NoReturn
if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from configobj import ConfigObj


class AcquisitionFunction:
    """Abstract class that defines the structure of a Acquisition Function
    """

    def __init__(self):
        self.context = None

    """Abstract class for the implementation of different Acquisition functions
    """
    def evaluate(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float) -> np.ndarray:
        """calculate the acquisition value given a sequence of mean (mu) and variance (sigma).
        For each entry in the input array they are combined into one output value.

        :param mu: array of mean values
        :param sigma: array of variance values
        :param incumbent: previously best target-evaluation
        :return: array of acquisition value
        """
        pass

    def single_expression(self, model,  mu: float, sigma: float, incumbent: float):
        """Expression Formulation of the acquisition function at one single point

        :param model: cplex model
        :param mu: mean prediction
        :param sigma: variance of the prediction
        :param incumbent: previously best evaluation
        :return: Acquisition function formulation expression
        """
        pass


    def _cdf(self, x: np.ndarray) -> np.ndarray:
        """helper method to calculate the cumulative distribution function"""
        return norm.cdf(x)

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        """helper method to calculate the probabilistic distribution function"""
        return norm.pdf(x)

    def set_context(self, context: 'Context') -> NoReturn:
        """add the context to he Acq. function so that i has access to process data in the context
        :param context: context instance
        """
        self.context = context

    def get_json(self) -> dict:
        json_dir = self.__dict__
        json_dir["class_name"] = self.__class__.__name__
        return json_dir

    def evaluate_inspector_acq(self, mu: np.ndarray, sigma: np.ndarray, context: 'Context', step: int) -> np.array:
        """secondary interface to evaluate the acquisition function post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: acq values
         """
        pass

    def evaluate_inspector_mu_sig(self, mu: np.ndarray, sigma: np.ndarray, context, step) -> np.array:
        """secondary interface to evaluate the uncertainty post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: mu and sigma estimates
         """
        pass

    @staticmethod
    def read_acq(config: 'ConfigObj', context) -> 'AcquisitionFunction':
        """reads the configuration from the config file and creates the acquisition function accordingly
        :param config: config parser instance
        :return: acquisitions function instance
        """
        from . import SUPPORTED_ACQS
        assert "function" in config, "config file must include 'Acquisition' section"
        from . import SUPPORTED_EXTENSIONS
        base_acq = SUPPORTED_ACQS[config["function"]].read_from_config(config)

        added_extensions = {}
        for key, extension in SUPPORTED_EXTENSIONS.items():
            if key in config:
                order = config[key].as_int("order")
                added_extensions[order] = {"key": key, "extension": extension}

        for i in sorted(added_extensions, reverse=True):
            base_acq = added_extensions[i]["extension"].read_from_config(config[added_extensions[i]["key"]], context, base_acq)
        return base_acq