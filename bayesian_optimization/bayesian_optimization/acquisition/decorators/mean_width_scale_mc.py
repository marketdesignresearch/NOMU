# Libs
import numpy as np

# Internal
from bayesian_optimization.acquisition.decorators.abstract_decorator import AbstractDecorator
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from configobj import ConfigObj


class MeanWidthScaledMC(AbstractDecorator):
    """Decorator to an acquisition function that scales the uncertainty
    according to the specified mean width budget.
    Mean width calculation with Monte Carlo Sampling
    """

    INSP_C = "mws_c"

    def __init__(
            self,
            acq_to_decorate,
            scale_mean_width: float,
            lower_bound: np.array,
            upper_bound: np.array,
            n_test_points: int,
            context: 'Context',
            once_only: bool = False,
    ):
        super().__init__(acq_to_decorate)
        self.scale_mean_width = scale_mean_width
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        assert len(self.lower_bound) == len(self.upper_bound), "lower and upper must have same dimensions"
        self.input_dim = len(self.lower_bound)
        self.n_test_points = n_test_points
        self.context = context
        self.test_points = np.random.uniform(low=self.lower_bound, high=self.upper_bound, size=(self.n_test_points, self.input_dim))
        self.c = None
        self.once_only = once_only

    def evaluate(self, mu: np.ndarray, sigma: np.ndarray, incumbent: float) -> np.ndarray:
        """calculate the acquisition value given a sequence of mean (mu) and variance (sigma).
        For each entry in the input array they are combined into one output value.

        :param mu: array of mean values
        :param sigma: array of variance values
        :param incumbent: previously best target-evaluation
        :return: array of acquisition value
        """
        if self.c is None or not self.once_only:
            self._calculate_scale()
        if self.context is not None and self.context.inspector is not None:
            self.context.inspector.dump_data(self.INSP_C, self.c)
        return self.acq_to_decorate.evaluate(mu, self.c*sigma, incumbent)

    def sigma_corrected(self, sigma: np.ndarray) -> np.array:
        """returns sigma scaled by the scaling factor c

        :param sigma: array of variance values
        :return: array of scaled variance values
        """
        return self.c * sigma

    def evaluate_inspector_mu_sig(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> Tuple[np.array, np.array]:
        """secondary interface to evaluate the uncertainty post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: mu and sigma estimates
         """
        if self.c is None or not self.once_only:
            self._calculate_scale()
        return self.acq_to_decorate.evaluate_inspector_mu_sig(mu, self.c*sigma, context, step)

    def evaluate_inspector_acq(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            context: 'Context',
            step: int
    ) -> np.array:
        """secondary interface to evaluate the acquisition function post experiment from the context

         :param mu: mean value
         :param sigma: variance value
         :param context: Context of the run
         :param step: bayesian optimization step
         :return: acq values
         """
        return self.evaluate(mu, sigma, 0.0)

    def _get_mean_uncertainty_width(self) -> float:
        """Calculate the mean width for the given gridpoints
        :return: mean width of the uncertainty
        """
        mean, sigma = self.context.estimator.regress(test_x=self.test_points)
        return 2*np.mean(sigma)

    def _scale_c(self, factor) -> NoReturn:
        """set scaling factor for sigma (named 'c') manually
        :param factor: factor which is used to scale sigma in the acquisition function
        """
        if self.c is None or not self.once_only:
            self._calculate_scale()
        self.c *= factor

    def _calculate_scale(self) -> NoReturn:
        """Calculate the the factor which the sigma has to be multiplied by to
        get de desired mean width. (desired mean width specified at construction of Acq. Func.)

        :return: factor for estimator to multiply sigma with
        """
        mean_width = self._get_mean_uncertainty_width()
        self.c = self.scale_mean_width / mean_width

    @staticmethod
    def read_from_config(config: 'ConfigObj', context: 'Context', base_acq: 'AcquisitionFunction') -> 'AcquisitionFunction':
        """reads the configuration from the config file and add the decorator configured accordingly to the give acq-function
        :param config: config parser instance
        :param context: context
        :param base_acq: Acquisition function wo wrap
        :return: acquisitions function instance
        """
        return MeanWidthScaledMC(
            acq_to_decorate=base_acq,
            scale_mean_width=config.as_float("scale_mean_width"),
            lower_bound=[float(i) for i in config.as_list("lower_bound")],
            upper_bound=[float(i) for i in config.as_list("upper_bound")],
            n_test_points=config.as_int("n_test_points"),
            context=context,
            once_only=config.as_bool("once_only"),
        )