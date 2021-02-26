import numpy as np
from bayesian_optimization.estimators.estimator import Estimator


class NNEstimator(Estimator):
    """Abstract level for Neural networks
    """

    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = epochs

    def estimate(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray,
            inspect: bool = True):
        pass
