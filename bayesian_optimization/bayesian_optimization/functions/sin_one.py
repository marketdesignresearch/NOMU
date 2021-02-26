import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


class SinOne(AbstractFunction):

    _sin_one = hpobench.SinOne()

    ORIGINAL_MIN_ARGUMENT = np.array(_sin_one.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[0.86743372]])
    ORIGINAL_UPPER_BOUNDS = np.array([1.])
    ORIGINAL_LOWER_BOUNDS = np.array([0.])
    ORIGINAL_MAX = _sin_one.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _sin_one.get_meta_information()["f_opt"]

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._sin_one.objective_function(x)["function_value"]

