import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


def a_i(i):
    return (i - 2.) / 2.



def Gfun(x, d_in):
    prod = 1
    for i in range(d_in):
        factor = (np.abs(4 * x[i] - 2.) + a_i(i+1)) / (1. + a_i(i+1))
        prod = prod * factor
    return prod


class GFunction2D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([[0.5, 1.]])
    ORIGINAL_MAX_ARGUMENT = np.array([[1., 1.], [0., 0.]])
    ORIGINAL_MAX = Gfun(ORIGINAL_MAX_ARGUMENT[0], 2)
    ORIGINAL_MIN = Gfun(ORIGINAL_MIN_ARGUMENT[0], 2)
    ORIGINAL_UPPER_BOUNDS = np.array([1., 1.])
    ORIGINAL_LOWER_BOUNDS = np.array([0., 0.])

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Gfun(np.array(x), 2)


class GFunction5D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.ones((1, 5))
    ORIGINAL_MIN_ARGUMENT[0, 0] = 0.5
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*5, [0.]*5])
    ORIGINAL_MAX = Gfun(ORIGINAL_MAX_ARGUMENT[0], 5)
    ORIGINAL_MIN = Gfun(ORIGINAL_MIN_ARGUMENT[0], 5)
    ORIGINAL_UPPER_BOUNDS = np.array([1.]*5)
    ORIGINAL_LOWER_BOUNDS = np.array([0.]*5)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Gfun(np.array(x), 5)


class GFunction10D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.ones((1, 10))
    ORIGINAL_MIN_ARGUMENT[0, 0] = 0.5
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*10, [0.]*10])
    ORIGINAL_MAX = Gfun(ORIGINAL_MAX_ARGUMENT[0], 10)
    ORIGINAL_MIN = Gfun(ORIGINAL_MIN_ARGUMENT[0], 10)
    ORIGINAL_UPPER_BOUNDS = np.array([1.]*10)
    ORIGINAL_LOWER_BOUNDS = np.array([0.]*10)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Gfun(np.array(x), 10)


class GFunction20D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.ones((1, 20))
    ORIGINAL_MIN_ARGUMENT[0, 0] = 0.5
    ORIGINAL_MAX_ARGUMENT = np.array([[1.]*20, [0.]*20])
    ORIGINAL_MAX = Gfun(ORIGINAL_MAX_ARGUMENT[0], 20)
    ORIGINAL_MIN = Gfun(ORIGINAL_MIN_ARGUMENT[0], 20)
    ORIGINAL_UPPER_BOUNDS = np.array([1.]*20)
    ORIGINAL_LOWER_BOUNDS = np.array([0.]*20)

    INVERT = True

    @classmethod
    def base_function(cls, x):
        return Gfun(np.array(x), 20)
