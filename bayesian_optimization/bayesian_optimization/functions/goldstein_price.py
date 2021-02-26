import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction



class GoldsteinPrice(AbstractFunction):

    _goldstein_price = hpobench.GoldsteinPrice()

    ORIGINAL_MIN = _goldstein_price.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_goldstein_price.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[2., 2.]])
    ORIGINAL_MAX = _goldstein_price.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_UPPER_BOUNDS = np.array([2., 2.])
    ORIGINAL_LOWER_BOUNDS = np.array([-2., -2.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._goldstein_price.objective_function(x)["function_value"]

