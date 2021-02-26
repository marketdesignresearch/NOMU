# Libs
import numpy as np

# Internals
from bayesian_optimization.acq_optimizer.extensions.callback import OptimizerCallback

# Type hints
from typing import *

if TYPE_CHECKING:
    from configobj import ConfigObj
    from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction


class DynamicC(OptimizerCallback):
    """Callback that adjusts the c-factor of the estimator which scales the sigma before feeding it into the acquisition
    function. If the found optima is within a epsilon bound around an existing sample then the c-factor is doubled
    and the therefore new acquisition function is once again optimized. the epsilon bound shrinks with increasing
    number of bayesian optimization steps. This is because the epsilon bound is defined by the starting epsilon
    value divided by the number of sample points.
    """

    INSP_ARG_MAX = "opt_arg_max"
    INSP_MU_ARG_MAX = "opt_mu_arg_max"
    INSP_SIGMA_ARG_MAX = "opt_sigma_arg_max"
    INSP_ACQ_ARG_MAX = "opt_acq_arg_max"
    INSP_MU = "opt_mu"
    INSP_SIGMA = "opt_sigma"
    INSP_ACQ = "opt_acq"
    INSP_C = "c"

    def __init__(
            self,
            x_range: float = 2.0,
            n_start_points: int = 8,
            range_fraction: float = 0.25,
            end_eps: float = 0.001,
            max_increase_iter: int = 10,
            n_steps: int = 15,
    ):
        """constructor
        :param x_range: range for the input values (default 2.0)
        :param n_start_points:number of initial samples (default 8)
        :param range_fraction: factor (default 0.25)
        :param end_eps: smallest epsilon (default 0.001)
        :param max_increase_iter: maximum number of tries to find new valid nex sample point (default 10)
        :param n_steps:number of stapes in the bo (default 15)
        """
        super().__init__()
        self.n_start_points = n_start_points
        self.range_fraction = range_fraction
        self.end_eps = end_eps
        self.x_range = x_range
        self.max_increase_iter = max_increase_iter
        self.start_eps = (self.x_range * self.range_fraction) / self.n_start_points
        self.n_steps = n_steps
        self.run_inspections = []

    def _in_eps(self, new_sample_x: np.array) -> bool:
        """check if the given potential next sample point is within the epsilon bound around the already sampled points
        :param new_sample_x: potential next sample point
        :return: whether the potential sample point is within any epsilon bound, or not
        """
        n_step = len(self.context.samples_x)-self.n_start_points
        step_eps = self.start_eps + (n_step * (self.end_eps - self.start_eps)/self.n_steps)
        dists = list(map(lambda x: np.sqrt(np.sum((x - new_sample_x) ** 2, axis=0)), self.context.samples_x))
        return any(list(map(lambda x: x < max(step_eps, self.end_eps), dists)))

    def run(
            self,
            new_sample_x: np.array,
            new_sample_mu: np.array,
            new_sample_sigma: np.array,
            new_sample_acq: np.array,
            mu: np.array,
            sigma: np.array,
            acq_values: np.array,
            iteration: int = 0
    ) -> Tuple[np.array, float, float, float, np.array, np.array, np.array]:
        """run the actual callback, checks if potential new sample is within epsilon distance to samples.
        if the next sample is within the epsilon distance then doubles the c factor of the acquisition function
        and then runs the optimization again. Note this means a recursive call.
        after the result is gotten the c factor in the acquisition function is halved again (for each of the recursions)
        so that the c factor is again at the starting value for the next step.

        :param new_sample_x: found optima which would be next sample in the optimizer
        :param new_sample_mu: mu value estimated for the found optima/ next sample
        :param new_sample_sigma: sigma value estimated for the found optima/ next sample
        :param new_sample_acq: acquisition value estimated for the found optima/ next sample
        :param mu: array of all mu evaluation needed for the optimization
        :param sigma: array of all sigma evaluation needed for the optimization
        :param acq_values: array of all acq evaluation needed for the optimization
        :param iteration: number denoting in which callback iteration we are in (max-depth of recursions)
        :return: potentially updated next sample, next samples mu estimates, next samples sigma estimates and next samples acq. value
        """
        if self._in_eps(new_sample_x) and iteration <= self.max_increase_iter:
            acq = self._find_mws_acq(self.context.acq)
            assert hasattr(acq, "_scale_c")

            acq._scale_c(2.)
            new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values = \
                self.context.acq_optimizer.optimize_wo_regress(
                    np.copy(self.context.samples_x),
                    np.copy(self.context.samples_y),
                    mu,
                    sigma
                )
            self.run_inspections.append(
                {
                    self.INSP_ARG_MAX: new_sample_x,
                    self.INSP_MU_ARG_MAX: new_sample_mu,
                    self.INSP_SIGMA_ARG_MAX: new_sample_sigma,
                    self.INSP_ACQ_ARG_MAX: new_sample_acq,
                    self.INSP_MU: mu,
                    self.INSP_SIGMA: sigma,
                    self.INSP_ACQ: acq_values,
                    self.INSP_C: acq.c
                }
            )
            iteration += 1
            result = self.run(
                new_sample_x,
                new_sample_mu,
                new_sample_sigma,
                new_sample_acq,
                mu,
                sigma,
                acq_values,
                iteration
            )
            acq._scale_c(0.5)
            return result
        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values

    @staticmethod
    def _find_mws_acq(acq: 'AcquisitionFunction'):
        """iterates over the decorator layers to find the mws, decorator
        :param acq: acquisition function (with decorator wrapped)
        :return: mws-decorator layer
        """
        for i in range(0, 10):
            if hasattr(acq, "acq_to_decorate") and not hasattr(acq, "_scale_c"):
                acq = acq.acq_to_decorate
            else:
                break
        return acq

    @classmethod
    def read_from_config(cls, config: 'ConfigObj') -> 'DynamicC':
        """reads the configuration of the DynamicC Callback from the given configfile and returns the
        accordingly created DynamicC Callback
        :param config: config parser instance
        :return: DynamicC Callback
        """
        return cls(
            x_range=config.as_float("x_range"),
            n_start_points=config.as_int("n_start_points"),
            range_fraction=config.as_float("range_fraction"),
            end_eps=config.as_float("end_eps"),
            max_increase_iter=config.as_int("max_increase_iter"),
            n_steps=config.as_int("n_steps"),
        )
