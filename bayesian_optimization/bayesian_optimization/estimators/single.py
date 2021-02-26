# Libs
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Model

# Internals
from bayesian_optimization.estimators.nn_estimator import NNEstimator

# Type hinting
from typing import *
from typing import NoReturn


class SingleMethod(NNEstimator):
    """NeuralNetwork estimator based on networks directly returning mean and uncertainty
    """

    def __init__(
            self,
            epochs: int,
    ):
        super().__init__(epochs)

    def fit(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray
    ) -> 'Model':
        """fit the neuronal model to the given sample data.
        if no specific batch_size if set in the Ensemble Method then the
        batch_size is set to be the same as the size of the sample set.
        :param samples_x: All input values of the samples
        :param samples_y: All target values of the samples
        :return: fitted model
        """
        self.context.nn_model.fit(x=samples_x, y=samples_y, epochs=self.epochs, verbose=0)
        if self.context.inspector and self.context.inspector.store_estimators:
            path = "{}/{}/{}".format(
                self.context.inspector.inspector_path,
                self.__class__.__name__,
                self.context.bo_step)
            os.makedirs(path, exist_ok=True)
            self.context.nn_model.save(path)
        return self.context.nn_model.model

    def get_model(self) -> 'Model':
        """return the NN-model
        :return: the network model
        """
        return self.context.nn_model.model

    def estimate(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray,
            inspect: bool = True
    ) -> [np.ndarray, np.ndarray]:
        """train the underlying model with the given samples and then
        get the estimation for mu and sigma for the test data

        :param samples_x: input values of the samples
        :param samples_y: target values of the samples
        :param test_x: data to estimate, for which mu and sigma should be calculated
        :param inspect: should the data be stored in the inspector
        :return: mu and sigma values for the test data
        """
        start_time = datetime.now()
        self.context.nn_model.fit(x=samples_x, y=samples_y, epochs=self.epochs, verbose=0)
        time_elapsed = datetime.now() - start_time
        mu, sigma = self.regress(test_x)
        if inspect:
            self._inspect(mu, sigma, time_elapsed)
            self._inspect_on_test_data()

        return mu, sigma

    def regress(self, test_x: np.ndarray) -> [np.ndarray, np.ndarray]:
        """only get the estimation for mu and sigma for the test data.
        Assumes that the underlying model is already trained

        :param test_x: data to estimate, for which mu and sigma should be calculated
        :return: mu and sigma values for the test data
        """
        self.context.nn_model.predict(test_x)
        return self.context.nn_model.predict(test_x)

    def _inspect(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            time_elapsed: int
    ):
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
            self.context.nn_model.save(path)

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
            res = self.regress(self.context.inspector.test_x)

            inspection_data["final_mu"] = res[0]
            inspection_data["final_sigma"] = res[1]
            inspection_data["final_acq"] = self.context.acq.evaluate(res[0], res[1], np.max(self.context.samples_y))
            self.context.inspector.add_estimation_test_data(inspection_data)


    @staticmethod
    def get_inspector_mu_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_mu"]

    @staticmethod
    def get_inspector_sigma_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_sigma"]

    @staticmethod
    def get_inspector_samples_x_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["samples_x"]

    @staticmethod
    def get_inspector_samples_y_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["samples_y"]

    @staticmethod
    def get_inspector_acq_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["final_acq"]

    def get_inspector_data_on_test_data(self, context, step):
        mus = context.inspector.estimations_on_test_data[step]["final_mu"]
        sigmas = context.inspector.estimations_on_test_data[step]["final_sigma"]
        samples_x = context.inspector.estimations_on_test_data[step]["samples_x"]
        samples_y = context.inspector.estimations_on_test_data[step]["samples_y"]
        incumbent = np.max(samples_y)
        acqs = context.acq.evaluate(mus, sigmas, incumbent)
        return samples_x, samples_y, mus, sigmas, acqs

    @staticmethod
    def _plot_inspector_mu_on_test_data(context, step, line_width=2):
        mus = SingleMethod.get_inspector_mu_on_test_data(context, step)
        plt.plot(context.inspector.test_x, mus, linewidth=line_width, label='mean', linestyle='-')

    @staticmethod
    def _plot_inspector_sigma_on_test_data(context, step, line_width=2):
        sigmas = SingleMethod.get_inspector_sigma_on_test_data(context, step)
        plt.plot(context.inspector.test_x, sigmas, color="red", linewidth=line_width, label='sigma',
                 linestyle=':')

    def load_model(self, base_path, step):
        """load the models which were generated earlier by the Inspector given to a Bayesian Optimization run
        :param base_path: base path to data of a BO run (where the context file is located)
        :param step: for which step the model should be loaded
        """
        path = "{}/{}/{}".format(
            base_path,
            self.__class__.__name__,
            step)
        model_name = os.path.basename(glob.glob(path + "/*.json")[0]).split(".json")[0]
        self.context.nn_model.load(path+"/"+model_name)
