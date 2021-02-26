# Libs
import numpy as np

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


class GridSearch(ACQOptimizer):
    """acquisition function optimizer
    By sampling the whole input space in a grid manner and rind the maxima of all these samples.
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

    """Implementation of a simple grid-search Optimizers that searches on a grid between lower and upper bounds
    """
    def __init__(
            self,
            lower_search_bounds: np.ndarray,
            upper_search_bounds: np.ndarray,
            n_gridpoints: int,
            callback: Union[None, OptimizerCallback] = None
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
        self.n_gridpoints = n_gridpoints
        self.gridpoints = None

    def optimize(
            self,
            sample_x: np.array,
            sample_y: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        """optimize the acquisition function defined in the context by
        sampling the the input space in a grid manner and evaluating the acquisition function
        for all these grid search.
        For the estimation of mu and sigma only a regression/prediction is run and no new fit.

        :param sample_x: input of samples
        :param sample_y: outputs of samples
        :return: input, target estimate, uncertainty estimate and acquisition function estimate for the new proposed
            sample as well as all (target, uncertainty) estimate and acq-values of the samples used for
            the acq-optimization
        """
        incumbent = np.max(sample_y)
        self._get_grid_points()
        mu, sigma = self.context.estimator.regress(self.gridpoints)
        acq_values = self.context.acq.evaluate(mu, sigma, incumbent)
        index = np.argmax(acq_values)
        new_sample_x = self.gridpoints[index]
        new_sample_mu = mu[index]
        new_sample_sigma = sigma[index]
        new_sample_acq = acq_values[index]
        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values

    def optimize_wo_regress(
            self,
            sample_x: np.array,
            sample_y: np.array,
            mu: np.array,
            sigma: np.array
    ) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array, np.array]:
        """finds the optima using grid search without prediction mu and sigma
        freshly but rather use the given mu and sigma values.

        :param sample_x: sample inputs
        :param sample_y: sample outputs
        :param mu: mu values of the gridpoints
        :param sigma: sigma values for the gridpoints
        :return: input, target estimate, uncertainty estimate and acquisition function estimate for the new proposed
            sample as well as all (target, uncertainty) estimate and acq-values of the samples used for
            the acq-optimization
        """
        incumbent = np.max(sample_y)
        acq_values = self.context.acq.evaluate(mu, sigma, incumbent)
        index = np.argmax(acq_values)
        new_sample_x = self.gridpoints[index]
        new_sample_mu = mu[index]
        new_sample_sigma = sigma[index]
        new_sample_acq = acq_values[index]
        return new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values

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
        new_sample_x, new_sample_mu, new_sample_sigma, new_sample_acq, mu, sigma, acq_values = self.optimize(sample_x, sample_y)
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
            self.context.inspector.dump_data(self.INSP_X, self.gridpoints)

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

    def _get_grid_points(self) -> NoReturn:
        """generate the grid-points for which the function is evaluated and for which the maximum evaluated.
        :return: grid-points as a mesh (can be multidimensional)
        """
        if self.gridpoints is None:
            grid_res = np.meshgrid(*[np.linspace(low, self.upper_search_bounds[index], self.n_gridpoints)
                                     for (index, low) in enumerate(self.lower_search_bounds)])
            self.gridpoints = np.array(list(zip(*np.vstack(map(np.ravel, grid_res)))))

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
        :param acq_values: acq value of the samples used for finding the acq-optima
        """
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
    ) -> 'GridSearch':
        """reads the configuration of the gridsearch from the given configfile and returns the
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
            n_gridpoints=config.as_int("n_test_points"),
            callback=callback
        )