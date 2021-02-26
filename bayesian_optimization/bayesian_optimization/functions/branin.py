import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


class Branin2D(AbstractFunction):

    _branin = hpobench.Branin()

    ORIGINAL_MAX_ARGUMENT = np.array([[-5, 0.]])
    ORIGINAL_MAX = _branin.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = np.array(_branin.get_meta_information()["f_opt"])
    ORIGINAL_MIN_ARGUMENT = np.array(_branin.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([10., 15.])
    ORIGINAL_LOWER_BOUNDS = np.array([-5., 0.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._branin.objective_function(x)["function_value"]


