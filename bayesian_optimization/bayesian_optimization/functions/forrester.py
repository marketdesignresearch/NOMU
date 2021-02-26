import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.acquisition.upperbound import UpperBound
from bayesian_optimization.functions.abstract_function import AbstractFunction


class Forrester(AbstractFunction):

    _forrester = hpobench.Forrester()

    ORIGINAL_MIN_ARGUMENT = np.array(_forrester.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[1.0]])
    ORIGINAL_UPPER_BOUNDS = np.array([1.])
    ORIGINAL_LOWER_BOUNDS = np.array([0.])
    ORIGINAL_MAX = _forrester.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _forrester.get_meta_information()["f_opt"]

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._forrester.objective_function(x)["function_value"]


