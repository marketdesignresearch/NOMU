import hpolib.benchmarks.synthetic_functions as hpobench
from hpolib.benchmarks.synthetic_functions.hartmann3 import Hartmann3
from hpolib.benchmarks.synthetic_functions.hartmann6 import Hartmann6
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock10D as RB10d
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


class Hartmann3D(AbstractFunction):

    _hartmann = hpobench.Hartmann3()

    ORIGINAL_MAX_ARGUMENT = np.array([[0., 0., 0.]])
    ORIGINAL_MAX = _hartmann.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _hartmann.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_hartmann.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([1., 1., 1.])
    ORIGINAL_LOWER_BOUNDS = np.array([0., 0., 0.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._hartmann.objective_function(x)["function_value"]


class Hartmann6D(AbstractFunction):

    _hartmann = hpobench.Hartmann6()

    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*6])
    ORIGINAL_MAX = _hartmann.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _hartmann.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_hartmann.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([1.]*6)
    ORIGINAL_LOWER_BOUNDS = np.array([0.]*6)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._hartmann.objective_function(x)["function_value"]


