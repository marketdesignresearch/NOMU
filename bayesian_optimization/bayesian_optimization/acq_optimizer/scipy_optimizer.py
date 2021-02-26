# Libs
import numpy as np
from scipy.optimize import minimize, Bounds

# Internal
from bayesian_optimization.acq_optimizer.acq_optimizer import ACQOptimizer
from bayesian_optimization.acq_optimizer.extensions.callback import OptimizerCallback
from bayesian_optimization.acq_optimizer.extensions.dynamic_c_callback import DynamicC
from bayesian_optimization.acq_optimizer.extensions.dynamic_c_exp import DynamicCExponential

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from configobj import ConfigObj


class SciPyOptimizer(ACQOptimizer):
    """Wrapper Class for optimizing function using scipy Optimizers to use as acquisition function.
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
            method: str = "Nelder-Mead"
    ) -> NoReturn:
        """constructor
        :param lower_search_bounds: lower bounds to the search input space
        :param upper_search_bounds: upper bounds to the search input space
        :param callback: callback instance to be run after the optimization
        :param method: name/identifier for the scipy-minimize method to use
        """
        super().__init__(
            lower_search_bounds=lower_search_bounds,
            upper_search_bounds=upper_search_bounds,
            callback=callback
        )
        self.method = method

    def optimize(
            self,
            sample_x: np.array,
            sample_y: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        """optimize the acquisition function defined in the context by using
        the scipy optimizer 'minimize'. For this an internal function is defined which returns the negative of the
        acquisition function and then the scipy minimize is applied to it.
        According to the method that is selected a global or local optimization scheme is used.

        :param sample_x: input of samples
        :param sample_y: outputs of samples
        :return: input, target estimate, uncertainty estimate and acquisition function estimate for the new proposed
            sample as well as all (target, uncertainty) estimate and acq-values of the samples used for
            the acq-optimization. Also the used input values are returned
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
            within_bounds = True
            for i, x_d in enumerate(x):
                if x_d < self.lower_search_bounds[i] or x_d > self.upper_search_bounds[i]:
                    within_bounds = False
            if not within_bounds:
                return 0
            m, s = self.context.estimator.regress([x])
            mu_f.append(m)
            sigma_f.append(s)
            a = self.context.acq.evaluate(m, s, inc)
            acq_f.append(a)
            x_f.append(x)
            return -a

        result = minimize(
            func,
            incumbent_x,
            args=(incumbent, mu, sigma, acq, xs),
            method=self.method,
            bounds=Bounds(self.lower_search_bounds, self.upper_search_bounds),
            options={"return_all": True, "fatol": 1e-8}
        )
        new_sample_x = result.x
        new_sample_mu, new_sample_sigma = self.context.estimator.regress([new_sample_x])
        print(result)
        new_sample_acq = result.fun
        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, result.allvecs, xs

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

    def get_optima(
            self,
            sample_x: np.array,
            sample_y: np.array,
            run_callback: bool = True,
    ) -> Tuple[np.array, float, float, float]:
        """Find the optima of the acquisition function based on the predictions
        which are based on the given samples.

        :param sample_x: x values of the samples (existing function evaluations)
        :param sample_y: target value of the samples (existing function evaluations)
        :param run_callback: whether to apply or not a potential optimization callback
        """
        self.context.estimator.fit(sample_x, sample_y)
        new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values, xs = self.optimize(sample_x,
                                                                                                             sample_y)
        if self.context.inspector is not None:
            self.context.inspector.dump_data(
                self.INSP_BASE,
                {
                    self.INSP_ARG_MAX: new_sample_x,
                    self.INSP_MU_ARG_MAX: new_sample_mu,
                    self.INSP_SIGMA_ARG_MAX: new_sample_sigma,
                    self.INSP_ACQ_ARG_MAX: new_sample_acq,
                    self.INSP_MU: mu,
                    self.INSP_SIGMA: sigma,
                    self.INSP_ACQ: acq_values,
                }
            )
            self.context.inspector.dump_data(self.INSP_X, xs)

        if run_callback and self.callback:
            new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values = self.callback.run(
                new_sample_x,
                new_sample_mu,
                new_sample_sigma,
                new_sample_acq,
                mu,
                sigma,
                acq_values,
                0
            )
            if self.context.inspector is not None:
                self.context.inspector.dump_data(
                    self.CALLBACK,
                    self.callback.run_inspections
                )
                self.callback.run_inspections = []
        self._inspect(new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values)
        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq

    def _inspect(
            self,
            arg_max: np.array,
            mu_arg_max: np.array,
            sigma_arg_max: np.array,
            acq_arg_max: np.array,
            mu: np.array,
            sigma: np.array,
            acq_values: np.array,
    ) -> NoReturn:
        """saves relevant data into the inspector specified in the context if the respected flags are set.
        :param arg_max: argument of the found optima
        :param mu_arg_max: respective mu value of the found optima of the acq. function
        :param sigma_arg_max: respective sigma value of the found optima of the acq. function
        :param acq_arg_max: acq value of the found optima of the acq. function
        :param mu: mu value of the samples used for finding the acq-optima
        :param sigma: sigma value of the samples used for finding the acq-optima
        :param acq_values: acq value of the samples used for finding the acq-optima        """
        if self.context.inspector is not None:
            self.context.inspector.dump_data(
                self.INSP_FINAL,
                {
                    self.INSP_ARG_MAX: arg_max,
                    self.INSP_MU_ARG_MAX: mu_arg_max,
                    self.INSP_SIGMA_ARG_MAX: sigma_arg_max,
                    self.INSP_ACQ_ARG_MAX: acq_arg_max,
                    self.INSP_MU: mu,
                    self.INSP_SIGMA: sigma,
                    self.INSP_ACQ: acq_values,
                }
            )

    @classmethod
    def read_from_config(
            cls,
            config: 'ConfigObj'
    ) -> 'SciPyOptimizer':
        """reads the configuration of the scipy-optimizer from the given configfile and returns the
        accordingly created grid search instance
        :param config: config parser instance
        :return: grid search instance
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
            method=config["method"],
            callback=callback
        )