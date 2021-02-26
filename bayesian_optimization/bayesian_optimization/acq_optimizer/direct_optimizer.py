# Libs
import sys
import os
import numpy as np
from scipydirect import minimize
import gc

# Internal
from bayesian_optimization.acq_optimizer.scipy_optimizer import SciPyOptimizer
from bayesian_optimization.acq_optimizer.extensions.callback import OptimizerCallback
from bayesian_optimization.acq_optimizer.extensions.dynamic_c_callback import DynamicC
from bayesian_optimization.acq_optimizer.extensions.dynamic_c_exp import DynamicCExponential

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from configobj import ConfigObj


class DirectOptimizer(SciPyOptimizer):
    """Class for the optimization of the acquisition function to find
    next point for evaluation. Is a wrapper Class for the scipydirect algorithm which
    is a python implementation of the DIRECT algorithm.
    """

    INSP_BASE = "opt_base_results"
    INSP_FINAL = "opt_final_results"

    CALLBACK = "opt_callback"

    INSP_X = "opt_grid"

    INSP_ARG_MAX = "opt_arg_max"
    INSP_MU_ARG_MAX = "opt_mu_arg_max"
    INSP_SIGMA_ARG_MAX = "opt_sigma_arg_max"
    INSP_ACQ_ARG_MAX = "opt_acq_arg_max"
    INSP_MU = "opt_mu"
    INSP_SIGMA = "opt_sigma"
    INSP_ACQ = "opt_acq"

    def __init__(
            self,
            lower_search_bounds: np.ndarray,
            upper_search_bounds: np.ndarray,
            callback: Union[None, OptimizerCallback] = None,
    ) -> NoReturn:
        """constructor
        :param lower_search_bounds: lower bounds to the search input space
        :param upper_search_bounds: upper bounds to the search input space
        :param callback: callback instance to be run after the optimization
        """
        super().__init__(
            lower_search_bounds=lower_search_bounds,
            upper_search_bounds=upper_search_bounds,
            callback=callback
        )

    def optimize(
            self,
            sample_x: np.array,
            sample_y: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        """finds the optima of the acquisition function. For that defines a internal function that evaluates
        the acquisition function and returns it's negative value. Then executes the scipydirect minimization
        algorithm to find the maxima of the acquisition function. While doing this keeps track of evaluated points

        :param sample_x: input of samples
        :param sample_y: outputs of samples
        :return: input value with maximal acquisition value, mu value for point with maximal acquisition value,
            sigma value for point with maximal acquisition value, maximal acquisition function value, all mu estimates,
            all sigma estimates, all acq. values evaluated, all input samples for the acquisition function
        """
        incumbent = np.max(sample_y)
        incumbent_x = sample_x[np.argmax(sample_y)]
        mu = []
        sigma = []
        acq = []
        xs = []

        def func(x: np.array, inc: float, mu_f: np.array, sigma_f: np.array, acq_f: np.array, x_f: np.array) -> float:
            """function that is internally used as it is passed to the minimize
            takes input and calculates the respective acquisition value and negates it as minimize is used
            but the original problem is maximization

            :param x: input value
            :param inc: incumbent
            :param mu_f: list of mu values to append mu-prediction to for persistence
            :param sigma_f: list of sigma values to append sigma-prediction to for persistence
            :param acq_f: list of acquisition function values to append for to persistence
            :param x_f: list of acquisition function sample inputs to append to for persistence
            :return:
            """
            m, s = self.context.estimator.regress(np.array([x]))
            mu_f.append(m)
            sigma_f.append(s)
            a = self.context.acq.evaluate(m, s, inc)
            acq_f.append(a)
            x_f.append(x)
            if len(x_f)%200 == 0:
                gc.collect()

            return -a

        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        result = minimize(
            func,
            args=(incumbent, mu, sigma, acq, xs),
            bounds=np.vstack((self.lower_search_bounds, self.upper_search_bounds)).T,
            maxT=100,
            maxf=5000
        )
        sys.stdout = old_stdout  # reset old stdout

        new_sample_x = result.x
        new_sample_mu, new_sample_sigma = self.context.estimator.regress(np.array([new_sample_x]))
        new_sample_acq = result.fun

        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq, xs

    def optimize_wo_regress(
            self,
            sample_x: np.array,
            sample_y: np.array,
            mu: np.array,
            sigma: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        """wrapper function for optimization version that does not require a new regression.

        :param sample_x: input of samples
        :param sample_y: outputs of samples
        :param mu: already regressed mu values
        :param sigma: already regressed sigma values
        :return: input, target estimate, uncertainty estimate and acquisition function estimate for the new proposed
            sample as well as all (target, uncertainty) estimate and acq-values of the samples used for
            the acq-optimization
        """
        return self.optimize(sample_x, sample_y)[:7]

    @classmethod
    def read_from_config(
            cls,
            config: 'ConfigObj'
    ) -> 'SciPyOptimizer':
        """reads the configuration of the DIRECT optimizer from the given configfile and returns the
        accordingly created grid search instance
        :param config: config parser instance
        :return: SciPyOptimizer instance
        """
        supported_callbacks = {
            "dynamic_c": DynamicC,
            "dynamic_c_exp": DynamicCExponential,
        }
        callback = None

        for key, extension in supported_callbacks.items():
            if key in config:
                callback = supported_callbacks[key].read_from_config(config[key])

        lower_bounds = np.array([float(i) for i in config.as_list("lower_search_bounds")])
        upper_bounds = np.array([float(i) for i in config.as_list("upper_search_bounds")])
        return cls(
            lower_search_bounds=lower_bounds,
            upper_search_bounds=upper_bounds,
            callback=callback
        )