# Libs
import numpy as np
from abc import ABC, abstractmethod


class AbstractFunction(ABC):
    """abstract class for a synthetic function
    """

    ORIGINAL_MAX = None
    ORIGINAL_MIN = None
    ORIGINAL_MIN_ARGUMENT = np.array([])
    ORIGINAL_MAX_ARGUMENT = np.array([])
    ORIGINAL_UPPER_BOUNDS = np.array([])
    ORIGINAL_LOWER_BOUNDS = np.array([])

    @property
    def INVERT(self) -> bool:
        """defines whether the original function as a minimization
        problem and thus has to be inverted or not
        :return: bool whether function has to be inverted to be maximization problem
        """
        raise NotImplementedError

    @staticmethod
    def base_function(x: np.array) -> float:
        """base implementation of the function
        :param x: input
        :return: function value of x
        """
        raise NotImplementedError

    @classmethod
    def original_to_scaled_y(cls, y: float) -> float:
        """scale output space to [-1.1)
        :param y: originl output
        :return: scaled output
        """
        return (y - (cls.ORIGINAL_MAX + cls.ORIGINAL_MIN) * 0.5) * 2. / (cls.ORIGINAL_MAX - cls.ORIGINAL_MIN)

    @classmethod
    def scaled_to_original_x(cls, x: np.array) -> np.array:
        """scale nomalized input from [-1.1)^d to original input space
        :param x: scaled input
        :return: original input
        """
        x_vec = []
        for i, x_i in enumerate(x):
            factor = (cls.ORIGINAL_UPPER_BOUNDS[i]-cls.ORIGINAL_LOWER_BOUNDS[i])/2.
            shift = (cls.ORIGINAL_UPPER_BOUNDS[i]+cls.ORIGINAL_LOWER_BOUNDS[i])/2.
            x_vec.append((x_i * factor) + shift)
        return np.array(x_vec)

    @classmethod
    def original_to_scaled_x(cls, x):
        """scale input space to [-1.1)^d
        :param x: original input
        :return: scaled input
        """
        x_vec = []
        for i, x_i in enumerate(x):
            factor = (cls.ORIGINAL_UPPER_BOUNDS[i]-cls.ORIGINAL_LOWER_BOUNDS[i])/2.
            shift = (cls.ORIGINAL_UPPER_BOUNDS[i]+cls.ORIGINAL_LOWER_BOUNDS[i])/2.
            x_vec.append((x_i - shift) / factor)
        return np.array(x_vec)

    @classmethod
    def original_to_scaled_x_arr(cls, x: np.array):
        """scale multiple inputs together
        :param x: original inputs
        :return: scaled inputs
        """
        return np.array([cls.original_to_scaled_x(x_i) for x_i in x])

    @classmethod
    def get_minima_y(cls) -> float:
        """get the minimum output from the scaled function
        :return: minimum function value (scaled)
        """
        if cls.INVERT:
            return -1. * cls.original_to_scaled_y(cls.base_function(cls.ORIGINAL_MAX_ARGUMENT[0]))
        return cls.original_to_scaled_y(cls.base_function(cls.ORIGINAL_MIN_ARGUMENT[0]))

    @classmethod
    def get_maxima_y(cls) -> float:
        """get the maximum output from the scaled function
        :return: maximum function value (scaled)
        """
        if cls.INVERT:
            return -1. * cls.original_to_scaled_y(cls.base_function(cls.ORIGINAL_MIN_ARGUMENT[0]))
        return cls.original_to_scaled_y(cls.base_function(cls.ORIGINAL_MAX_ARGUMENT[0]))

    @classmethod
    def get_minima_x(cls) -> np.array:
        """get the argument to the minimum output from the scaled function
        :return: minimum function argument (scaled)
        """
        if cls.INVERT:
            return np.array([cls.original_to_scaled_x(extrema) for extrema in cls.ORIGINAL_MAX_ARGUMENT])
        return np.array([cls.original_to_scaled_x(extrema) for extrema in cls.ORIGINAL_MIN_ARGUMENT])

    @classmethod
    def get_maxima_x(cls) -> np.array:
        """get the argument to the maximum output from the scaled function
        :return: maximum function argument (scaled)
        """
        if cls.INVERT:
            return np.array([cls.original_to_scaled_x(extrema) for extrema in cls.ORIGINAL_MIN_ARGUMENT])
        return np.array([cls.original_to_scaled_x(extrema) for extrema in cls.ORIGINAL_MAX_ARGUMENT])

    @classmethod
    def evaluate_scaled(cls, x_values: np.array) -> np.array:
        """evaluate the scaled function
        :param x_values: input values
        :return: scaled function evaluations
        """
        if cls.INVERT:
            return np.array([[-1 * cls.original_to_scaled_y(cls.base_function(cls.scaled_to_original_x(x)))] for x in x_values])
        return np.array([[cls.original_to_scaled_y(cls.base_function(cls.scaled_to_original_x(x)))] for x in x_values])

    @classmethod
    def evaluate(cls, x_values: np.array) -> np.array:
        """evaluate the original function
        :param x_values: input values
        :return: original function evaluations
        """
        return [cls.base_function(x) for x in x_values]

    @classmethod
    def get_regret(cls, y_max_found) -> float:
        """calcluate the regret for a given output value (scaled)
        :param y_max_found: optimal value found
        :return: regret between optimal found and true optima
        """
        y_opt = cls.evaluate_scaled(cls.get_maxima_x())[0]
        return np.abs(y_max_found - y_opt)
