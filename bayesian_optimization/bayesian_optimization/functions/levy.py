import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


def w_i(x, i):
    return 1 + ((x[i] - 1)/4)


def Lev(x, d_in):
    b = np.power((w_i(x, d_in - 1) - 1), 2) * (1 + np.power(np.sin(2*np.pi*w_i(x, d_in-1)), 2))

    res = np.power(np.sin(np.pi * w_i(x, 0)), 2)
    for i in range(0, d_in-1):
        a = np.power((w_i(x, i) - 1), 2) * (1+10*np.power(np.sin(np.pi*w_i(x, i)+1), 2))
        res += a + b
    return res


class Levy(AbstractFunction):

    _levy = hpobench.Levy()

    ORIGINAL_MIN_ARGUMENT = np.array(_levy.get_meta_information()["optima"])
    ORIGINAL_MAX_ARGUMENT = np.array([[-14.037]])
    ORIGINAL_UPPER_BOUNDS = np.array([10.])
    ORIGINAL_LOWER_BOUNDS = np.array([-15.])
    ORIGINAL_MAX = _levy.objective_function(ORIGINAL_MAX_ARGUMENT[0])["function_value"]
    ORIGINAL_MIN = _levy.get_meta_information()["f_opt"]

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return cls._levy.objective_function(x)["function_value"]


class Levy5D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([[-10]*5])
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*5])
    ORIGINAL_MAX = Lev(ORIGINAL_MAX_ARGUMENT[0], 5)
    ORIGINAL_MIN = Lev(ORIGINAL_MIN_ARGUMENT[0], 5)
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*5)
    ORIGINAL_LOWER_BOUNDS = np.array([-10.]*5)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Lev(np.array(x), 5)


class Levy10D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([[-10]*10])
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*10])
    ORIGINAL_MAX = Lev(ORIGINAL_MAX_ARGUMENT[0], 10)
    ORIGINAL_MIN = Lev(ORIGINAL_MIN_ARGUMENT[0], 10)
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*10)
    ORIGINAL_LOWER_BOUNDS = np.array([-10.]*10)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Lev(np.array(x), 10)


class Levy20D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([[-10]*20])
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*20])
    ORIGINAL_MAX = Lev(ORIGINAL_MAX_ARGUMENT[0], 20)
    ORIGINAL_MIN = Lev(ORIGINAL_MIN_ARGUMENT[0], 20)
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*20)
    ORIGINAL_LOWER_BOUNDS = np.array([-10.]*20)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Lev(np.array(x), 20)
