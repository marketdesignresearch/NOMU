# Libs
import numpy as np

# Internal
from bayesian_optimization.acq_optimizer.acq_optimizer import ACQOptimizer

# Type hinting
from typing import *


class NNACQOptimizer(ACQOptimizer):
    """Abstract class for shared functionality between different NN specific optimizers
    """

    def __init__(
            self,
            lower_search_bounds: np.ndarray,
            upper_search_bounds: np.ndarray
    ):
        super().__init__(lower_search_bounds, upper_search_bounds)

    def get_optima(
            self,
            sample_x: np.ndarray,
            sample_y: np.ndarray,
            run_callback: bool = True
    ) -> Tuple[np.array, float, float, float]:
        """Find the optima of the acquisition function based on the predictions
        which are based on the given samples.

        :param sample_x: x values of the samples (existing function evaluations)
        :param sample_y: target value of the samples (existing function evaluations)
        :param run_callback: whether to apply or not a potential optimization callback
        """
        pass

