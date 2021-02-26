import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


beta_def = .5

def Perm(x, d_in):
    outer = 0
    for ii in range(1, d_in + 1):
        inner = 0
        for jj in range(1, d_in + 1):
            xj = x[jj - 1]
            inner = inner + (jj ** ii + beta_def) * ((xj / jj) ** ii - 1)
        outer = outer + inner ** 2
    return outer


class Perm2D(AbstractFunction):

    ORIGINAL_MAX = 110.5
    ORIGINAL_MIN = 0.0
    ORIGINAL_MIN_ARGUMENT = np.array([[1., 2.]])
    ORIGINAL_MAX_ARGUMENT = np.array([[-2., -2.]])
    ORIGINAL_UPPER_BOUNDS = np.array([2., 2.])
    ORIGINAL_LOWER_BOUNDS = np.array([-2., -2.])

    BETA = 0.5

    INVERT = True

    @classmethod
    def base_function(cls, x):
        x1 = x[0]
        x2 = x[1]
        return ((1 + cls.BETA) * (x1 - 1) + (2 + cls.BETA) * (x2 / 2 - 1)) ** 2 + (
                    (1 + cls.BETA) * (x1 ** 2 - 1) + (2 ** 2 + cls.BETA) * ((x2 / 2) ** 2 - 1)) ** 2


class Perm5D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([[1., 2., 3., 4., 5.]])
    ORIGINAL_MAX_ARGUMENT = np.array([[-5., -5., -5., -5., -5.]])
    ORIGINAL_MAX = Perm(ORIGINAL_MAX_ARGUMENT[0], 5)
    ORIGINAL_MIN = Perm(ORIGINAL_MIN_ARGUMENT[0], 5)
    ORIGINAL_UPPER_BOUNDS = np.array([5., 5., 5., 5., 5.])
    ORIGINAL_LOWER_BOUNDS = np.array([-5., -5., -5., -5., -5.])


    @classmethod
    def base_function(cls, x):
        return Perm(np.array(x), 5)

class Perm10D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([range(1,11)])
    ORIGINAL_MAX_ARGUMENT = np.array([[-10.]*10])
    ORIGINAL_MAX = Perm(ORIGINAL_MAX_ARGUMENT[0], 10)
    ORIGINAL_MIN = Perm(ORIGINAL_MIN_ARGUMENT[0], 10)
    ORIGINAL_UPPER_BOUNDS = np.array([10.]*10)
    ORIGINAL_LOWER_BOUNDS = np.array([-10.]*10)


    @classmethod
    def base_function(cls, x):
        return Perm(np.array(x), 10)

class Perm20D(AbstractFunction):

    ORIGINAL_MIN_ARGUMENT = np.array([range(1,21)])
    ORIGINAL_MAX_ARGUMENT = np.array([[-20.]*20])
    ORIGINAL_MAX = Perm(ORIGINAL_MAX_ARGUMENT[0], 20)
    ORIGINAL_MIN = Perm(ORIGINAL_MIN_ARGUMENT[0], 20)
    ORIGINAL_UPPER_BOUNDS = np.array([20.]*20)
    ORIGINAL_LOWER_BOUNDS = np.array([-20.]*20)


    @classmethod
    def base_function(cls, x):
        return Perm(np.array(x), 20)


def main():
    pass


if __name__ == "__main__":
    main()
