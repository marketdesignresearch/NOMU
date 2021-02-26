# Libs
import numpy as np

# Internals
from bayesian_optimization.utils.utils import get_grid_points_multidimensional

# type hints
from typing import *
if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context


class Estimator:
    """Abstract class that defines the structure of an estimator. Can be implemented in various ways.
    For example for A EnsembleMethod, A sample based Method or a single Netor based method.
    """

    def __init__(self):
        self.context = None

    def set_context(self, context: 'Context'):
        """set the context to the estimator to have two way communication
        :param context:
        :return:
        """
        self.context = context

    def get_mean(self, samples_x: np.ndarray, samples_y: np.ndarray, test_x: np.ndarray):
        pass

    def get_uncertainty(self, samples_x: np.ndarray, samples_y: np.ndarray, test_x: np.ndarray):
        pass

    def get_bounds(self, samples_x: np.ndarray, samples_y: np.ndarray, test_x: np.ndarray):
        pass

    def estimate(self, samples_x: np.ndarray, samples_y: np.ndarray, test_x: np.ndarray, inspect: bool = True):
        pass

    def fit(self, samples_x: np.ndarray, samples_y: np.ndarray):
        pass

    def regress(self, test_x: np.array) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_json(self):
        return {
            "method": self.__class__.__name__
        }

    def get_mean_uncertainty_width(self, lower_bound: np.array, upper_bound: np.array, n_test_points: int) -> float:
        """calculate the mean width between lower and upper bound. The uncertainty width is taken on the
        given number of test points and then added up and divided by the number of test points.
        :param lower_bound: Lower bounds of the space which should be considered for the mean width calculation
        :param upper_bound: Upper bounds of the space which should be considered for the mean width calculation
        :param n_test_points: number of equidistant test points (per dimension) for the mean width calculation
        :return: mean width for the given input space.
        """
        gridpoints = get_grid_points_multidimensional(lower_bound, upper_bound, n_test_points)
        mean, sigma = self.regress(test_x=gridpoints)
        return 2*np.mean(sigma)

    def load_model(self, base_path, step):
        pass

