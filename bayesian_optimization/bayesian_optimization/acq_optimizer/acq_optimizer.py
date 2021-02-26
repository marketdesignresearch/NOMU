# Libs
import numpy as np

# Internal
from bayesian_optimization.acq_optimizer.extensions.callback import OptimizerCallback

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from configobj import ConfigObj
    from bayesian_optimization.context.context import Context


class ACQOptimizer:
    """Abstract class for the optimization of the acquisition function to find
    next point for evaluation
    """

    def __init__(
            self,
            lower_search_bounds: np.ndarray,
            upper_search_bounds: np.ndarray,
            callback: Union[None, OptimizerCallback] = None
    ) -> NoReturn:
        """constructor
        :param lower_search_bounds: lower bounds to the search input space
        :param upper_search_bounds: upper bounds to the search input space
        :param callback: callback instance to be run after the optimization
        """
        self.context = None
        self.lower_search_bounds = lower_search_bounds
        self.upper_search_bounds = upper_search_bounds
        self.callback = callback

    def set_context(
            self,
            context: 'Context'
    ) -> NoReturn:
        """assigns the context to the class instance to allow two way communication
        :param context: context instance
        """
        self.context = context
        if self.callback:
            self.callback.set_context(self.context)

    def get_optima(
            self,
            sample_x: np.ndarray,
            sample_y: np.ndarray,
            run_callback: bool = True
    ) -> Tuple[np.array, float, float, float]:
        """calculate the optima of the given acquisition function

        :param sample_x: input of samples
        :param sample_y: outputs of samples
        :param run_callback: should callback be executed or not
        :return: x-value of the optima, y-value of the optima and the acquisition function value of the optima
        """

    @staticmethod
    def read_acq_optimizer(
            config: 'ConfigObj'
    ) -> 'ACQOptimizer':
        """reads the configuration of the Acq. Optimizer from the given configfile and returns the
        accordingly created Acq. Optimizer
        :param config: config parser instance
        :return: acquisition function optimizer instance
        """
        from bayesian_optimization.acq_optimizer import SUPPORTED_ACQ_OPTIMIZERS
        assert "optimizer" in config, "config file must include 'Acquisition' section"
        return SUPPORTED_ACQ_OPTIMIZERS[config["optimizer"]].read_from_config(config)
