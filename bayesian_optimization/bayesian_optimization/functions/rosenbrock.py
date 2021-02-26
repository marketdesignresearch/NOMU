import hpolib.benchmarks.synthetic_functions as hpobench
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock5D as RB5d
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock10D as RB10d
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock20D as RB20d
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


class Rosenbrock2D(AbstractFunction):

    _rosenbrock = hpobench.Rosenbrock()

    ORIGINAL_MAX = 1102581
    ORIGINAL_MIN = _rosenbrock.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_rosenbrock.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[10., -5.]])
    ORIGINAL_UPPER_BOUNDS = np.array([10., 10.])
    ORIGINAL_LOWER_BOUNDS = np.array([-5., -5.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._rosenbrock.objective_function(x)["function_value"]


class Rosenbrock5D(AbstractFunction):

    _rosenbrock = RB5d()

    ORIGINAL_MAX_ARGUMENT = np.array([[10., 10., 10., 10.,  10.]])
    ORIGINAL_MAX = _rosenbrock.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _rosenbrock.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_rosenbrock.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([10., 10., 10., 10., 10.])
    ORIGINAL_LOWER_BOUNDS = np.array([-5., -5., -5, -5, -5])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._rosenbrock.objective_function(x)["function_value"]

class Rosenbrock10D(AbstractFunction):

    _rosenbrock = RB10d()

    ORIGINAL_MAX_ARGUMENT = np.array([[10.]*10])
    ORIGINAL_MAX = _rosenbrock.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _rosenbrock.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_rosenbrock.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*10)
    ORIGINAL_LOWER_BOUNDS = np.array([-5.]*10)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._rosenbrock.objective_function(x)["function_value"]

class Rosenbrock20D(AbstractFunction):

    _rosenbrock = RB20d()

    ORIGINAL_MAX_ARGUMENT = np.array([[10.]*20])
    ORIGINAL_MAX = _rosenbrock.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _rosenbrock.get_meta_information()["f_opt"]
    ORIGINAL_MIN_ARGUMENT = np.array(_rosenbrock.get_meta_information()["optima"])
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*20)
    ORIGINAL_LOWER_BOUNDS = np.array([-5.]*20)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._rosenbrock.objective_function(x)["function_value"]

