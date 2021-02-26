# Libs
import glob
import os
from datetime import datetime
import numpy as np
import tqdm
from tensorflow.keras.models import Model

# Internals
from bayesian_optimization.estimators.nn_estimator import NNEstimator

# Type hinting
from typing import *
from typing import NoReturn


class SampleMethod(NNEstimator):
    """NeuralNetwork estimator based on sampling multiple networks
    """

    def __init__(
            self,
            epochs: int,
            base_l2_reg: float,
            n_samples=500,
            normalize_regularization=True
    ):
        super().__init__(epochs)
        self.n_samples = n_samples
        self.base_l2_reg = base_l2_reg
        self.normalize_regularization = normalize_regularization

    def get_regluarization_factor(self):
        """calculates the appropriate regularization factor for the estimator based on the normalized
        regularization factor specified in the config file
        :return: regularization factor
        """
        if self.normalize_regularization:
            if self.context.nn_model.dropout:
                return (1.-self.context.nn_model.dropout) / len(self.context.samples_x)
            return 1./len(self.context.samples_x)
        return 1.0

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
        self.context.nn_model.l2reg = self.get_regluarization_factor() * self.base_l2_reg
        self.context.nn_model.fit(x=samples_x, y=samples_y, epochs=self.epochs, verbose=0)
        if self.context.inspector and self.context.inspector.store_estimators:
            path = "{}/{}/{}".format(
                self.context.inspector.inspector_path,
                self.__class__.__name__,
                self.context.bo_step)
            os.makedirs(path, exist_ok=True)
            self.context.nn_model.save(path)
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

    def regress(
            self,
            test_x: np.ndarray
    ) -> [np.ndarray, np.ndarray]:
        """only get the estimation for mu and sigma for the test data.
        Assumes that the underlying model is already trained

        :param test_x: data to estimate, for which mu and sigma should be calculated
        :return: mu and sigma values for the test data
        """
        y_pred_list = []
        #for i in tqdm.tqdm(range(self.n_samples)):
        for i in range(self.n_samples):
            y_pred = self.context.nn_model.predict(test_x)
            y_pred_list.append(y_pred)
        y_preds = np.concatenate(y_pred_list, axis=1)
        y_mean = np.asarray([[m] for m in np.mean(y_preds, axis=1)]).reshape(-1, 1)
        y_sigma = np.asarray([[s] for s in np.std(y_preds, axis=1)]).reshape(-1, 1)
        return y_mean, y_sigma


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
                "model_mu_estimations": [],
                "final_mu": None,
                "final_sigma": None,
                "final_acq": None,
                "samples_x": np.copy(self.context.samples_x),
                "samples_y": np.copy(self.context.samples_y),
            }
            y_pred_list = []
            for i in tqdm.tqdm(range(self.n_samples)):
                y_pred = self.context.nn_model.predict(self.context.inspector.test_x)
                y_pred_list.append(y_pred)
                inspection_data["model_mu_estimations"].append(y_pred)
            y_preds = np.concatenate(y_pred_list, axis=1)
            y_mean = np.asarray([[m] for m in np.mean(y_preds, axis=1)]).reshape(-1, 1)
            y_sigma = np.asarray([[s] for s in np.std(y_preds, axis=1)]).reshape(-1, 1)
            inspection_data["final_mu"] = y_mean
            inspection_data["final_sigma"] = y_sigma
            inspection_data["final_acq"] = self.context.acq.evaluate(y_mean, y_sigma, np.max(self.context.samples_y))
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
