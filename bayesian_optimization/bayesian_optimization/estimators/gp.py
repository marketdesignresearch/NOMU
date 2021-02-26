# Libs
import os
import sys
from datetime import datetime
import dill as pickle
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Internals
from bayesian_optimization.estimators.estimator import Estimator
from bayesian_optimization.kernel.kernel import Kernel
from bayesian_optimization.utils.utils import config_list_int_or_none

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from configobj import ConfigObj


# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=UserWarning)


class GP(Estimator):
    """Wrapper class for the Gaussian Proccess.
    Uses the Sklearn implementation of the Gaussian Process.
    """

    def __init__(
            self,
            kernel=C(
                constant_value=10,
                constant_value_bounds=(1, 1000)
                ) * RBF(length_scale=1, length_scale_bounds=(1e-3, 2)),
            alpha=1e-7,
            optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=10,
            normalize_y=True,
            copy_X_train=True,
            random_state=None,
            std_min=0.,
            kernel_once=False,
    ):
        super().__init__()
        self.std_min = std_min
        self.kernel = kernel
        self.gp = GaussianProcessRegressor(
            kernel=kernel.kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state
        )
        self.kernel_once = kernel_once

    @ignore_warnings(category=ConvergenceWarning)
    def get_mean(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray
    ) -> np.ndarray:
        """ runs the gp estimator to fit the data and evaluate the mean estimate for the test data

        :param samples_x: All input values of the samples
        :param samples_y: All target values of the samples
        :param test_x: test inputs to evaluate the means on
        :return: mean predictions for the test x
        """
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.gp.fit(samples_x, samples_y)
        sys.stdout = old_stdout
        return self.gp.predict(test_x)

    @ignore_warnings(category=ConvergenceWarning)
    def get_mean_and_cov(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray
    ) -> List[np.ndarray]:
        """ runs the gp estimator to fit the data and evaluate the mean estimate and covariance for the test data

        :param samples_x: All input values of the samples
        :param samples_y: All target values of the samples
        :param test_x: test inputs to evaluate the means on
        :return: mean and covariance for the test x
        """
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.gp.fit(samples_x, samples_y)
        sys.stdout = old_stdout
        mean, cov = self.gp.predict(test_x, return_cov=True)
        return [mean, cov]

    @ignore_warnings(category=ConvergenceWarning)
    def fit(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray
    ) -> 'NoReturn':
        """fit the the given sample data.
        :param samples_x: All input values of the samples
        :param samples_y: All target values of the samples
        """
        params = None
        if not self.context.bo_step == 0 and self.kernel_once:
            params = self.gp.kernel_.get_params()
            params = self.gp.kernel.fix_params(params=params)
            self.gp.kernel = self.gp.kernel.set_params(**params)
            self.gp.kernel = self.gp.kernel_.set_params(**params)
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.gp.fit(samples_x, samples_y)
        sys.stdout = old_stdout
        if self.context.inspector and self.context.inspector.store_estimators:
            path = "{}/{}/{}".format(
                self.context.inspector.inspector_path,
                self.__class__.__name__,
                self.context.bo_step)
            os.makedirs(path, exist_ok=True)
            with open(path + "/gp.pickle", "wb") as f:
                pickle.dump(self.gp, f)

    @ignore_warnings(category=ConvergenceWarning)
    def estimate(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray,
            inspect: bool = True
    ) -> Tuple[np.array, np.array]:
        """train the underlying model with the given samples and then
        get the estimation for mu and sigma for the test data

        :param samples_x: input values of the samples
        :param samples_y: target values of the samples
        :param test_x: data to estimate, for which mu and sigma should be calculated
        :param inspect: should the data be stored in the inspector
        :return: mu and sigma values for the test data
        """
        start_time = datetime.now()
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        self.gp.fit(samples_x, samples_y)

        time_elapsed = datetime.now() - start_time
        mean, sigma = self.regress(test_x)
        sys.stdout = old_stdout
        if inspect:
            self._inspect(mean, sigma, time_elapsed)
            self._inspect_on_test_data()
        return mean, sigma

    @ignore_warnings(category=ConvergenceWarning)
    def regress(
            self,
            test_x: np.ndarray
    ) -> [np.ndarray, np.ndarray]:
        """only get the estimation for mu and sigma for the test data.
        Assumes that the underlying model is already trained

        :param test_x: data to estimate, for which mu and sigma should be calculated
        :return: mu and sigma values for the test data
        """
        mus, sigmas = self.gp.predict(test_x, return_std=True)
        return mus, np.array([[x+self.std_min] for x in sigmas])

    @ignore_warnings(category=ConvergenceWarning)
    def _inspect(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            time_elapsed: int
    ) -> NoReturn:
        """create a dictionary containing various interesting information for inspection
        :param mu: estimated mu
        :param sigma: estimated sigma
        :param time_elapsed: time spent for the estimation
        """
        if self.context.inspector and self.context.inspector.inspect_estimation:
            inspection_data = {
                "estimator": self.__class__.__name__,
                "final_mu": mu,
                "final_sigma": sigma,
                "time_elapsed": time_elapsed,
                "samples_x": np.copy(self.context.samples_x),
                "samples_y": np.copy(self.context.samples_y),
            }
            self.context.inspector.add_estimation(inspection_data)
        if self.context.inspector and self.context.inspector.store_estimators:
            path = "{}/{}/{}".format(
                self.context.inspector.inspector_path,
                self.__class__.__name__,
                self.context.bo_step)
            os.makedirs(path, exist_ok=True)
            with open(path + "/gp.pickle", "wb") as f:
                pickle.dump(self.gp, f)

    @ignore_warnings(category=ConvergenceWarning)
    def _inspect_on_test_data(self) -> NoReturn:
        """run the estimator on syntetic test data
        :return:
        """
        if self.context.inspector and self.context.inspector.estimate_test_data:
            inspection_data = {
                "estimator": self.__class__.__name__,
                "final_mu": None,
                "final_sigma": None,
                "final_acq": None,
                "samples_x": np.copy(self.context.samples_x),
                "samples_y": np.copy(self.context.samples_y),
            }
            mus, sigmas = self.regress(self.context.inspector.test_x)
            inspection_data["final_mu"] = mus
            inspection_data["final_sigma"] = sigmas
            inspection_data["final_acq"] = self.context.acq.evaluate(mus, sigmas, np.max(self.context.samples_y),
                                                                     inspect=False)

            self.context.inspector.add_estimation_test_data(inspection_data)

    """
    @staticmethod
    def get_inspector_mu_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_mu"]

    @staticmethod
    def get_inspector_sigma_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_sigma"]

    @staticmethod
    def get_inspector_samples_x_on_test_data(context, step):
        print(context.inspector.estimations_on_test_data)
        return context.inspector.estimations_on_test_data[step]["samples_x"]

    @staticmethod
    def get_inspector_samples_y_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["samples_y"]

    @staticmethod
    def get_inspector_acq_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_acq"]
    """

    def load_model(self, base_path: str, step: int = None, structured: bool = False) -> NoReturn:
        """ Load saved model
        :param base_path: basepath of the save model file
        :param step: step number, needed for stuctured loading
        :param structured: specify if file is loaded from the base_path or from base_path/GP/step/gp.pickle
        :return:
        """
        if not structured:
            with open(base_path, "rb") as f:
                self.gp = pickle.load(f)
        else:
            path = "{}/{}/{}".format(
                base_path,
                self.__class__.__name__,
                step)
            with open(path + "/gp.pickle", "rb") as f:
                self.gp = pickle.load(f)

    @staticmethod
    def read_from_config(config: 'ConfigObj') -> NoReturn:
        """read the config file and construct the GP instance accordingly
        :param config: config object defining the object
        :return:
        """
        kernel = Kernel.read_from_config(config["Kernel"])
        return GP(
            kernel=kernel,
            alpha=config.as_float("alpha"),
            optimizer=config["optimizer"],
            n_restarts_optimizer=config.as_int("n_restarts_optimizer"),
            normalize_y=config.as_bool("normalize_y"),
            copy_X_train=config.as_bool("copy_X_train"),
            random_state=config_list_int_or_none(config, "random_state"),
            std_min=config.as_float("std_min"),
            kernel_once=config.as_bool("kernel_once")
        )