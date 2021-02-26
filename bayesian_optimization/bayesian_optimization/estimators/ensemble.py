# Libs
import glob
import os
import time
from datetime import datetime
import numpy as np

# Internals
from bayesian_optimization.estimators.nn_estimator import NNEstimator

# Type hinting
from typing import *
from typing import NoReturn

class EnsembleMethod(NNEstimator):
    """class that defines the estimation process of an Ensemble Method
    which creates an ensemble of trained neural networks and evaluates that
    ensemble to then return a mean and uncertainty estimate.
    """

    def __init__(
            self,
            n_ensembles: int,
            epochs=500,
            batch_size=None,
            global_seed=0,
            random_seed=False,
            normalize_regularization=True
    ):
        """constructor
        :param n_ensembles: number of networks in the ensemble
        :param epochs: number of epochs to train each net with
        :param batch_size: batchsize for the training of the network
        """
        super().__init__(epochs)
        self.n_ensembles = n_ensembles
        self.models = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_seed = global_seed
        self.normalize_regularization = normalize_regularization
        if random_seed:
            self.global_seed = int(time.time())

    def get_regluarization_factor(self) -> float:
        """calculates the appropriate regularization factor for the estimator based on the normalized
        regularization factor specified in the config file
        :return: regularization factor
        """
        if self.normalize_regularization:
            return 1./len(self.context.samples_x)
        return 1.0

    def fit(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray
    ) -> NoReturn:
        """fit the neuronal model to the given sample data.
        if no specific batch_size if set in the Ensemble Method then the
        batch_size is set to be the same as the size of the sample set.
        :param samples_x: All input values of the samples
        :param samples_y: All target values of the samples
        """
        self.models = []
        batch = self.batch_size
        if not batch:
            batch = len(samples_x)
        for i in range(0, self.n_ensembles):
            self.global_seed = self.global_seed + i
            new_model = self.context.nn_model.create_copy(self.global_seed, self.get_regluarization_factor())
            new_model.set_context(self.context)
            new_model.model.compile()
            new_model.fit(x=samples_x, y=samples_y, epochs=self.epochs, verbose=0, batch_size=batch)
            self.models.append(new_model)
            # print("Fit Models {}/{} - {}".format(i+1, self.n_ensembles, new_model.model_name))
        if self.context.inspector and self.context.inspector.store_estimators:
            for i, model in enumerate(self.models):
                path = "{}/{}/{}".format(
                    self.context.inspector.inspector_path,
                    self.__class__.__name__,
                    self.context.bo_step)
                os.makedirs(path, exist_ok=True)
                model.save(path)

    def estimate(
            self,
            samples_x: np.ndarray,
            samples_y: np.ndarray,
            test_x: np.ndarray,
            inspect: bool = True
    ) -> Tuple[np.array, np.array]:
        """train the  underlying model with the given samples and then
        get the estimation for mu and sigma for the test data

        :param samples_x: input values of the samples
        :param samples_y: target values of the samples
        :param test_x: data to estimate, for which mu and sigma should be calculated
        :param inspect: should the data be stored in the inspector
        :return: mu and sigma values for the test data
        """
        start_time = datetime.now()
        self.fit(samples_x, samples_y, verbose=0)
        time_elapsed = datetime.now() - start_time
        mean, sigma = self.regress(test_x)
        if inspect:
            self._inspect(mean, sigma, time_elapsed)
            self._inspect_on_test_data()
        return mean, sigma

    def regress(
            self,
            test_x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """only get the estimation for mu and sigma for the test data.
        Assumes that the underlying model is already trained

        :param test_x: data to estimate, for which mu and sigma should be calculated
        :return: mu and sigma values for the test data
        """

        number_of_networks = len(self.models)
        sum_mu = np.zeros((len(test_x), 1))
        sum_mu_squared = np.zeros((len(test_x), 1))
        sum_var = np.zeros((len(test_x), 1))
        for i, model in enumerate(self.models):
            mu, sigma = self.predict_single_model(model, test_x)
            sum_mu += mu
            sum_mu_squared += mu ** 2
            if sigma is not None:
                sum_var += sigma
            #print("Regress Models {}/{}".format(i+1, len(self.models)))

        mu_pred = sum_mu / number_of_networks
        var_pred = (sum_var + sum_mu_squared) / number_of_networks - (mu_pred) ** 2
        return mu_pred, var_pred

    @staticmethod
    def predict_single_model(
            model, test_x
    ) -> Tuple[Union[None,np.ndarray], Union[None,np.ndarray]]:
        """get the mu and r prediction for one single model in the ensemble
        :param model: the nn-Model for which the prediction should be made
        :param test_x: data to estimate, for which mu and sigma should be calculated
        :return: mu and sigma values for the test data
        """
        pred_res = model.predict(test_x)
        if model.no_noise:
            return pred_res, None
        return pred_res[:, 0].reshape(-1, 1), pred_res[:, 1].reshape(-1, 1)

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
                "number_of_networks": len(self.models),
                "final_mu": mu,
                "final_sigma": sigma,
                "time_elapsed": time_elapsed,
                "samples_x": np.copy(self.context.samples_x),
                "samples_y": np.copy(self.context.samples_y),
            }
            self.context.inspector.add_estimation(inspection_data)

        if self.context.inspector and self.context.inspector.store_estimators:
            for i, model in enumerate(self.models):
                path = "{}/{}/{}".format(
                    self.context.inspector.inspector_path,
                    self.__class__.__name__,
                    self.context.bo_step)
                os.makedirs(path, exist_ok=True)
                model.save(path)

    def _inspect_on_test_data(self) -> NoReturn:
        """run the estimator on syntetic test data
        :return:
        """
        if self.context.inspector and self.context.inspector.estimate_test_data:
            number_of_networks = len(self.models)
            inspection_data = {
                "estimator": self.__class__.__name__,
                "number_of_networks": number_of_networks,
                "model_mu_estimations": [],
                "model_sigma_estimations": [],
                "model_acq_estimations": [],
                "model_weights": [],
                "final_mu": None,
                "final_sigma": None,
                "final_acq": None,
                "samples_x": np.copy(self.context.samples_x),
                "samples_y": np.copy(self.context.samples_y),
            }
            sum_mu = np.zeros((len(self.context.inspector.test_x), 1))
            sum_mu_squared = np.zeros((len(self.context.inspector.test_x), 1))
            sum_var = np.zeros((len(self.context.inspector.test_x), 1))
            for model in self.models:
                mu, sigma = self.predict_single_model(model, self.context.inspector.test_x)
                sum_mu += mu
                sum_mu_squared += mu ** 2
                if sigma is not None:
                    sum_var += sigma
                    inspection_data["model_sigma_estimations"].append(sigma)

                inspection_data["model_mu_estimations"].append(mu)
                inspection_data["model_weights"].append(model.model.get_weights())
            mu_pred = sum_mu / number_of_networks
            var_pred = (sum_var + sum_mu_squared) / number_of_networks - (sum_mu / number_of_networks) ** 2
            inspection_data["final_mu"] = mu_pred
            inspection_data["final_sigma"] = var_pred
            inspection_data["final_acq"] = self.context.acq.evaluate(mu_pred, var_pred, np.max(self.context.samples_y))
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

    @staticmethod
    def get_inspector_model_mus_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["model_mu_estimations"]

    @staticmethod
    def get_inspector_model_sigmas_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["model_sigma_estimations"]

    @staticmethod
    def get_inspector_model_acqs_on_test_data(context, step):
        return context.inspector.estimations_on_test_data[step]["model_acq_estimations"]

    def load_model(self, base_path, step) -> NoReturn:
        """load the models which were generated earlier by the Inspector given to a Bayesian Optimization run
        :param base_path: base path to data of a BO run (where the context file is located)
        :param step: for which step the model should be loaded
        """
        path = "{}/{}/{}".format(
            base_path,
            self.__class__.__name__,
            step)
        self.models = []
        self.global_seed = 0
        for i, m in enumerate(glob.glob(path+"/*.h5")):
            self.global_seed = self.global_seed + i
            m_name = m.split(".h5")[0]
            new_model = self.context.nn_model.create_copy(self.global_seed)
            self.models.append(new_model.load(m_name))


