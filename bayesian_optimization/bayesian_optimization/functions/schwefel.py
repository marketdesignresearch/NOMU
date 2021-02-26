import hpolib.benchmarks.synthetic_functions as hpobench
import numpy as np
from bayesian_optimization.functions.abstract_function import AbstractFunction


def single_eval(x):
    s = [x_i * np.sin(np.sqrt(np.abs(x_i))) for x_i in x]
    return 418.9829 * 3 - np.sum(s)

class Schwefel3D(AbstractFunction):

    c = 418.9829
    x_opt = 420.9687
    d = 3

    @classmethod
    def base_function(cls, x):
        s = [x_i * np.sin(np.sqrt(np.abs(x_i))) for x_i in x]
        return cls.c * cls.d - np.sum(s)

    ORIGINAL_MAX_ARGUMENT = np.array([[-x_opt]*3])
    ORIGINAL_MAX = single_eval(ORIGINAL_MAX_ARGUMENT[0])
    ORIGINAL_MIN_ARGUMENT = np.array([[x_opt]*3])
    ORIGINAL_MIN = single_eval(ORIGINAL_MIN_ARGUMENT[0])
    ORIGINAL_UPPER_BOUNDS = np.array([500., 500., 500.])
    ORIGINAL_LOWER_BOUNDS = np.array([-500., -500., -500.])
    INVERT = True




