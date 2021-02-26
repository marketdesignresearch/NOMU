import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


class Camelback2D(AbstractFunction):

    _camelback = hpobench.Camelback()

    ORIGINAL_MIN = _camelback.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_camelback.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[-3, -2.], [3, 2.]])
    ORIGINAL_MAX = _camelback.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_UPPER_BOUNDS = np.array([3., 2.])
    ORIGINAL_LOWER_BOUNDS = np.array([-3., -2.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._camelback.objective_function(x)["function_value"]



