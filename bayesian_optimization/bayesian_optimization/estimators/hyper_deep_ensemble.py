# Libs
import glob
import os
import time
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

# Internals
from bayesian_optimization.scores.scores import gaussian_nll_score, mse_score
from bayesian_optimization.estimators.hyper_param_ensemble import HyperParamEnsemble
from bayesian_optimization.context.context import Context
from bayesian_optimization.estimators.nn_estimator import NNEstimator

# Type hinting
from typing import *


class HyperDeepEnsembleMethod(NNEstimator):
    """class that defines the estimation process of an Ensemble Method
    which creates an ensemble of trained neural networks and evaluates that
    ensemble to then return a mean and uncertainty estimate.
    """

    def __init__(
            self,
            context: Context,
            epochs: int = 500,
            batch_size=None,
            global_seed: int = 0,
            random_seed: bool = False,
            normalize_regularization: bool = True,
            test_size: float = 0.2,
            kappa: int = 50,
            K: int = 5,
            score=gaussian_nll_score,
            score_single_model=mse_score,
            upper_bound_on_non_unique_models: int = 50,
            random_state: Optional[int] = None,
            dropout_probability_range: Tuple = (0.001, 0.9),
            l2reg_range: Tuple = (10**-3, 10**3),
            fixed_row_init: bool = True,
    ):
        """constructor
        :param n_ensembles: number of networks in the ensemble
        :param epochs: number of epochs to train each net with
        :param batch_size: batchsize for the training of the network
        """
        super().__init__(epochs)
        self.models = []
        self.weighted_models = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.global_seed = global_seed
        self.normalize_regularization = normalize_regularization
        self.kappa = kappa
        self.K = K
        self.score = score,
        self.score_single_model = score_single_model,
        self.upper_bound_on_non_unique_models = upper_bound_on_non_unique_models
        self.test_size = test_size
        self.random_state = random_state or global_seed
        self.dropout_probability_range = dropout_probability_range
        self.l2reg_range = l2reg_range
        self.fixed_row_init = fixed_row_init
        if random_seed:
            self.global_seed = int(time.time())
        self.HPE = HyperParamEnsemble(
            context=context,
            base_model=context.nn_model,
            kappa=kappa,
            test_size=test_size,
            K=K,
            epochs=epochs,
            batch_size=batch_size,
            fixed_row_init=fixed_row_init,
            dropout_probability_range=dropout_probability_range,
            l2reg_range=l2reg_range,
            score_single_model=score_single_model,
            score=score,
            global_seed=global_seed,
            upper_bound_on_non_unique_models=upper_bound_on_non_unique_models,
        )


    @staticmethod
    def _update_seed(seed: Union[int, None], add: int) -> Union[int, None]:
        """update the given seed. Adds the given value specified under 'add' to the given seed
        If the given seen is None this method will return None as well
        :param seed: input see
        :param add: value to be added to the seed
        :return: new seed
        """
        return [None if seed is None else seed + add][0]

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
        self.weighted_models = []
        self.HPE.hyper_deep_ens(samples_x, samples_y)

        model_names = [e["modelname"] for e in self.HPE.model_dicts]
        distinct = []

        for model_dict in self.HPE.model_dicts:
            if model_dict["modelname"] not in distinct:
                distinct.append(model_dict["modelname"])
                self.weighted_models.append({
                    "model": model_dict["model"],
                    "weight": model_names.count(model_dict["modelname"]),
                })

        # Todo save model
        # if self.context.inspector and self.context.inspector.store_estimators:

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
        self.fit(samples_x, samples_y)
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

        sum_mu = np.zeros((len(test_x), 1))
        sum_mu_squared = np.zeros((len(test_x), 1))
        sum_var = np.zeros((len(test_x), 1))
        for i, weight_dict in enumerate(self.weighted_models):
            model = weight_dict["model"]
            weight = weight_dict["weight"]
            w = weight/len(self.weighted_models)
            mu, sigma = self.predict_single_model(model, test_x)
            sum_mu += mu*w
            sum_mu_squared += (mu ** 2)*w
            if sigma is not None:
                sum_var += sigma*w
            # print("Regress Models {}/{}".format(i+1, len(self.weighted_models)))

        mu_pred = sum_mu
        var_pred = (sum_var + sum_mu_squared) - (mu_pred) ** 2
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

