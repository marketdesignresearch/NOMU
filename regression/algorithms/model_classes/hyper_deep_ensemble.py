# -*- coding: utf-8 -*-
"""
This file contains the model class HyperDeepEnsembles
"""

# Libs
from collections import OrderedDict
from itertools import product
import os
import re
from datetime import datetime
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.initializers import RandomUniform  # , GlorotUniform
from sklearn.model_selection import train_test_split
import pickle
from typing import NoReturn, Union, List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from collections import Counter
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s, update_seed
from algorithms.losses import gaussian_nll
from algorithms.callbacks import PredictionHistory_DE
from algorithms.custom_activation_functions import softplus_wrapper

# %% Class for Deep Ensembles Approach


class HyperDeepEnsemble:

    """
    Hyper Deep Ensembles (DE).

    ...

    Attributes
    ----------
    models : OrderedDict
        Tf.keras models.
    parameters : OrderedDict
        Parametersdict for each model.
    histories : OrderedDict
        Training histories for each model.
    scaler_input: StandardScaler
        Scaling input to mean=0, std=1
    scaler_target: StandardScaler
        Scaling target to mean=0, std=1

    Methods
    -------
    set_parameters()
        Sets the parameters for each model.
    initialize_models()
        Initializes the neural network architectures for all models.
    compile_models()
        Compiles the models.
    fit_models()
        Fits the models on specified dataset (x,y), where x is the input and y the output.
    predict()
        Predicts the output for a given input x for all models.
    calculate_mean_std()
        Calculates the mean and the uncertainty at a given input x for all models.
    plot_histories()
        Plots training histories for all models.
    reset_attributes()
        Resets the attributes
    save_models()
        Saves all models and relevant informations.
    load_models()
        Loads models specified by model_numbers.

    """

    def __init__(self) -> NoReturn:

        """Constructor of the class UBNN."""

        # Attributes
        self.parameters = OrderedDict()
        self.models = OrderedDict()
        self.models_counts = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []
        self.variable_hyperparameters_values = OrderedDict()
        self.current_seed_counter = OrderedDict()
        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()

    def set_parameters(
        self,
        layers: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        epochs: Union[int, Tuple[int, int], List[int]],
        batch_size: Union[int, Tuple[int, int], List[int]],
        l2reg: Union[float, Tuple[float, float], List[float]],  # hyper possible
        optimizer_name: Union[str, List[str]],
        seed_init: Union[int, List[int]],
        loss: Union[str, List[str]],
        dropout_prob: Union[float, Tuple[float, float], List[float]],  # hyper possible
        K: Union[int, List[int]],
        kappa: Union[int, List[int]],
        test_size: Union[int, List[int]],
        stratify: Union[bool, List[bool]],
        fixed_row_init: Union[bool, List[bool]],
        refit: Union[bool, List[bool]],
        softplus_min_var: Union[float, List[float]],
        optimizer_learning_rate: Optional[List[float]] = None,
        optimizer_clipnorm: Optional[List[float]] = None,
        optimizer_momentum: Optional[List[float]] = None,
        optimizer_nesterov: Optional[List[bool]] = None,
        optimizer_beta_1: Optional[List[float]] = None,
        optimizer_beta_2: Optional[List[float]] = None,
        optimizer_epsilon: Optional[List[float]] = None,
        optimizer_amsgrad: Optional[List[bool]] = None,
        normalize_data: Optional[Union[bool, List[bool]]] = False,
    ) -> NoReturn:

        # TODO: update description
        """Sets the attributes of the class HyperDeepEnsemble.

        Arguments (Note: for each argument also multiple can be specified at once using a list)
        ----------
        layers : list
            Layers of neural network architectures for "y-architecture".
            e.g., [2,5,10,20,1] denotes a 2-hidden-layer NN with input dim 4, output dim 1 and
            5, 10 and 20 hidden nodes (=units), respectively.
        epochs : int
            Epochs in training.
        batch_size: integer
            Batch Size in training.
        l2reg : int
            L2 weights (and bias) regularization.
        optimizer_name: str
            Name of the optimizer for gradient descent updates.
        seed_init : int
            Seed for weight initialization.
        loss : str
            Loss function for training the deep ensembles. Currently, 'nll' for negative log likelihood
            which incroporates data noise or 'mse' when data noise is assumed to be zero.
        number_of_networks: int
            Number of neural networks in the ensemble.
        softplus_min_var : float
            Minimum data noise when 'nll' is used for numerical stability.
        optimizer_learning_rate : float
            SGD or Adam parameter
        optimizer_clipnorm : int
            SGD or Adam parameter
        optimizer_momentum : float
            SGD parameter
        optimizer_nesterov : bool
            SGD parameter
        optimizer_beta_1 : float
            Adam parameter
        optimizer_beta_2 : float
            Adam parameter
        optimizer_epsilon : float
            Adam parameter
        optimizer_amsgrad : bool
            Adam parameter
        normalize_data : bool
                If true, data is normalized s.t. mean=0, std=1

        """

        # optimizer
        if optimizer_clipnorm is None:
            optimizer_clipnorm = 1000  # default: clipnorm to a large value
        if optimizer_name == "SGD":
            if optimizer_learning_rate is None:
                optimizer_learning_rate = [0.01]  # default: 0.01
            if optimizer_momentum is None:
                optimizer_momentum = 0.0  # default: 0.0
            if optimizer_nesterov is None:
                optimizer_nesterov = False  # default: False
        elif optimizer_name == "Adam":
            if optimizer_learning_rate is None:
                optimizer_learning_rate = [0.001]  # default: learning_rate=0.001
            if optimizer_beta_1 is None:
                optimizer_beta_1 = 0.9  # default: beta_1=0.9
            if optimizer_beta_2 is None:
                optimizer_beta_2 = 0.999  # default: beta_2=0.999
            if optimizer_epsilon is None:
                optimizer_epsilon = 1e-07  # default: epsilon=1e-07
            if optimizer_amsgrad is None:
                optimizer_amsgrad = False  # default: amsgrad=False
        else:
            raise NotImplementedError(
                "{} not implemented as optimizer.".format(optimizer_name)
            )
        # set parameters
        parameter_keys = [
            "layers",
            "epochs",
            "actual_epochs",
            "batch_size",
            "l2reg",
            "seed_init",
            "loss",
            "dropout_prob",
            "K",
            "kappa",
            "test_size",
            "stratify",
            "fixed_row_init",
            "refit",
            "softplus_min_var",
            "optimizer",
            "learning_rate",
            "momentum",
            "nesterov",
            "beta_1",
            "beta_2",
            "epsilon",
            "amsgrad",
            "clipnorm",
            "normalize_data",
        ]
        if not isinstance(layers, list):
            layers = [layers]
        if not isinstance(epochs, list):
            epochs = [epochs]
        actual_epochs = [0]
        if not isinstance(batch_size, list):
            batch_size = [batch_size]
        if not isinstance(l2reg, list):
            l2reg = [l2reg]
        if not isinstance(seed_init, list):
            seed_init = [seed_init]
        if not isinstance(loss, list):
            loss = [loss]
        if not isinstance(dropout_prob, list):
            dropout_prob = [dropout_prob]
        if not isinstance(K, list):
            K = [K]
        if not isinstance(kappa, list):
            kappa = [kappa]
        if not isinstance(test_size, list):
            test_size = [test_size]
        if not isinstance(stratify, list):
            stratify = [stratify]
        if not isinstance(fixed_row_init, list):
            fixed_row_init = [fixed_row_init]
        if not isinstance(refit, list):
            refit = [refit]
        if not isinstance(softplus_min_var, list):
            softplus_min_var = [softplus_min_var]
        if not isinstance(optimizer_name, list):
            optimizer_name = [optimizer_name]
        if not isinstance(optimizer_learning_rate, list):
            optimizer_learning_rate = [optimizer_learning_rate]
        if not isinstance(optimizer_momentum, list):
            optimizer_momentum = [optimizer_momentum]
        if not isinstance(optimizer_nesterov, list):
            optimizer_nesterov = [optimizer_nesterov]
        if not isinstance(optimizer_beta_1, list):
            optimizer_beta_1 = [optimizer_beta_1]
        if not isinstance(optimizer_beta_2, list):
            optimizer_beta_2 = [optimizer_beta_2]
        if not isinstance(optimizer_epsilon, list):
            optimizer_epsilon = [optimizer_epsilon]
        if not isinstance(optimizer_amsgrad, list):
            optimizer_amsgrad = [optimizer_amsgrad]
        if not isinstance(optimizer_clipnorm, list):
            optimizer_clipnorm = [optimizer_clipnorm]
        if not isinstance(normalize_data, list):
            normalize_data = [normalize_data]
        parameters_values = list(
            product(
                layers,
                epochs,
                actual_epochs,
                batch_size,
                l2reg,
                seed_init,
                loss,
                dropout_prob,
                K,
                kappa,
                test_size,
                stratify,
                fixed_row_init,
                refit,
                softplus_min_var,
                optimizer_name,
                optimizer_learning_rate,
                optimizer_momentum,
                optimizer_nesterov,
                optimizer_beta_1,
                optimizer_beta_2,
                optimizer_epsilon,
                optimizer_amsgrad,
                optimizer_clipnorm,
                normalize_data,
            )
        )
        parameters = [OrderedDict(zip(parameter_keys, x)) for x in parameters_values]
        self.model_keys = [
            "Hyper_Deep_Ensemble_{}".format(i + 1) for i in range(len(parameters))
        ]
        # Set Attributes
        i = 0
        for key in self.model_keys:
            self.parameters[key] = parameters[i]
            self.models[key] = None
            self.models_counts[key] = None
            self.histories[key] = OrderedDict()
            self.variable_hyperparameters_values = OrderedDict()
            self.current_seed_counter[key] = parameters[i]["seed_init"]
            i += 1

    def hyper_deep_ens(
        self,
        x: np.array,
        y: np.array,
        score,
        score_single_model,
        random_state: Optional[int] = None,
        s: float = 0.05,
        activation: str = "relu",
        verbose: int = 0,
    ) -> NoReturn:
        # TODO: add decsription

        for ensemble_key in self.model_keys:

            if self.parameters[ensemble_key]["normalize_data"]:
                print("Fit function: Fit & Transform x-train...")
                self.scaler_input.fit(x)
                x = self.scaler_input.transform(x)
                print("Fit function: Fit & Transform y-train...")
                y = np.array(y).reshape(-1, 1)
                self.scaler_target.fit(y)
                y = self.scaler_target.transform(y)

            print()
            print(f"BUILD HYPER DEEP ENSEMBLE: {ensemble_key}")
            print(
                "--------------------------------------------------------------------------"
            )
            if verbose > 0:
                pretty_print_dict(self.parameters[ensemble_key])

            # full data
            self.x = x
            self.y = y

            # split once train/test
            test_size = self.parameters[ensemble_key]["test_size"]
            self.xtr, self.xtest, self.ytr, self.ytest = train_test_split(
                x, y, test_size=test_size, random_state=random_state
            )
            print(
                f"\nSplit: Train:{1-test_size}->Shape:{self.xtr.shape}/Test:{test_size}->Shape:{self.xtest.shape} with random_state:{random_state}"
            )
            # initialize & compile hyper deep ensemble via random search -> sets self.models[ensemble_key]
            print()
            print("######################")
            print("1.RANDOM_SEARCH")
            print("######################")
            self._random_search(
                ensemble_key=ensemble_key, s=s, activation=activation, verbose=verbose
            )
            print()
            print("######################")
            print("2. HYPER_ENS")
            print("######################")
            current_ensemble = self._hyper_ens(
                ensemble_key=ensemble_key,
                score=score,
                score_single_model=score_single_model,
                verbose=verbose,
                forbidden_fit_keys=[],
            )

            if self.parameters[ensemble_key]["stratify"]:
                # keep only UNIQUE ones after first _hyper_ens
                self.models[ensemble_key] = OrderedDict(current_ensemble)
                print()
                print("######################")
                print("a. STRATIFY")
                print("######################")
                current_ensemble = self._stratify(
                    ensemble_key=ensemble_key,
                    s=s,
                    activation=activation,
                    verbose=verbose,
                )
                self.models[ensemble_key] = current_ensemble
                # only fit the newly stratified models, since the others (1st row) already have been fit
                forbidden_fit_keys = [
                    x
                    for x in self.models[ensemble_key].keys()
                    if self._get_model_key_index(x)[0] == 1
                ]
                print()
                print("######################")
                print("b. HYPER_ENS (again)")
                print("######################")
                current_ensemble = self._hyper_ens(
                    ensemble_key=ensemble_key,
                    score=score,
                    score_single_model=score_single_model,
                    verbose=verbose,
                    forbidden_fit_keys=forbidden_fit_keys,
                )
                del forbidden_fit_keys

            # FINAL DICT OF COUNTS OF UNIQUE MODELS
            self.models_counts[ensemble_key] = self._calculate_dict_of_duplicates(
                current_ensemble
            )
            # FINAL DICT OF UNIQUE MODELS
            self.models[ensemble_key] = OrderedDict(current_ensemble)
            # UPDATE ATTRIBUTES
            self._update_attributes(
                ensemble_key=ensemble_key, valid_keys=self.models[ensemble_key].keys()
            )
            # PRINT FINAL ENSEMBLE
            self.print_final_ensemble_info(ensemble_key)

            del current_ensemble

            # REFIT (optionally) on all trainig points with adjusted l2reg parameters
            if self.parameters[ensemble_key]["refit"]:
                print("######################")
                print("3. REFIT ALL NNs")
                print("######################")
                l2reg_factor = self.xtr.shape[0] / self.x.shape[0]
                if verbose > 0:
                    print(f"new_l2reg=l2reg*{l2reg_factor}")
                new_layer_parameters = deepcopy(self.models_counts)
                for k, v in new_layer_parameters.items():
                    for k1, v1 in v.items():
                        new_layer_parameters[k][k1] = {
                            "l2reg": self.variable_hyperparameters_values[k][
                                f"NN_1-{self._get_model_key_index(k1)[1]}"
                            ]["l2reg"]
                            * l2reg_factor
                        }
                self._refit_all_models(
                    new_layer_parameters=new_layer_parameters, verbose=verbose
                )

    def _refit_all_models(
        self,
        verbose: int = 0,
        new_layer_parameters: Optional[dict] = None,
    ) -> NoReturn:
        # TODO: add description
        for ensemble_key in self.model_keys:

            # set new layer parameters (CURRENTLY ONLY SUPPORTS UPDATE OF REGULARIZER)
            if new_layer_parameters is not None:
                for model_key, model in self.models[ensemble_key].items():
                    if verbose > 0:
                        print(model_key)
                    parameter_dict = new_layer_parameters.get(ensemble_key, {}).get(
                        model_key
                    )
                    if parameter_dict is None:
                        continue
                    for k, v in parameter_dict.items():
                        if k == "l2reg":
                            for layer in model.layers:
                                for attr in ["bias_regularizer", "kernel_regularizer"]:
                                    if hasattr(layer, attr):
                                        if verbose > 1:
                                            print(
                                                f"old:{getattr(layer, attr).get_config()}"
                                            )
                                        setattr(layer, attr, l2(v))
                                        if verbose > 1:
                                            print(
                                                f"new:{getattr(layer, attr).get_config()}\n"
                                            )

            self._compile_models(ensemble_key=ensemble_key)
            self._fit_models(
                ensemble_key=ensemble_key,
                x=self.x,
                y=self.y,
                forbidden_fit_keys=[],
                verbose=verbose,
            )

    def print_final_ensemble_info(self, ensemble_key: Optional[str] = None) -> NoReturn:
        # TODO: add description

        if ensemble_key is not None:
            self._print_helper(ensemble_key)
        else:
            for ensemble_key in self.model_keys:
                self._print_helper(ensemble_key)

    def predict(
        self,
        x: np.array,
        verbose: int = 0,
    ) -> Dict[str, List[np.array]]:

        """Predicts the output for each model on a input point x. In the case of Hyper Deep Ensembles potentially with WEIGHTED ensembles.

        Arguments
        ----------
        x :
            input data (features).

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_2]])
            {'Hyper_Deep_Ensemble_1':[array([[mean_1],[mean_2]], dtype=float32),
                                    array([[std_1],[std_2]], dtype=float32)],
             'Hyper_Deep_Ensemble_2':...
            }

        """

        predictions = OrderedDict()
        for ensemble_key, ensemble in self.models.items():
            p = self.parameters[ensemble_key]

            if p["normalize_data"]:
                print("Prediction function: Transform x-test...")
                if len(x.shape) == 1:  # if 1d, format to 2d array for transformation
                    x = x.reshape(-1, 1)
                x = self.scaler_input.transform(x)

            if verbose > 0:
                print(self.models_counts[ensemble_key])
            total_number_of_models = sum(self.models_counts[ensemble_key].values())

            if p["loss"] == "nll":
                # see https://en.wikipedia.org/wiki/Mixture_distribution#Moments
                sum_mu_w = np.zeros((len(x), 1))
                sum_mu_squared_w = np.zeros((len(x), 1))
                sum_var_w = np.zeros((len(x), 1))
                for model_key, model in ensemble.items():
                    w = (
                        self.models_counts[ensemble_key][model_key]
                        / total_number_of_models
                    )  # individual model weight
                    tmp = model.predict(x)[:, 0].reshape(-1, 1)
                    sum_mu_w += tmp * w
                    sum_mu_squared_w += (tmp) ** 2 * w
                    sum_var_w += model.predict(x)[:, 1].reshape(-1, 1) * w
                    if verbose > 0:
                        print(f"weight:{w}")
                        print(f"prediction:{tmp}")
                        print(f"sum_mu_w:{sum_mu_w}")
                        print(f"sum_mu_squared_w:{sum_mu_squared_w}")
                        print(f"sum_var_w:{sum_var_w}")
                # calculate weighted mean of mixture model
                mu_pred = sum_mu_w

                # calculate weighted std (ddof==0 version)
                std_pred = np.sqrt((sum_var_w + sum_mu_squared_w) - (sum_mu_w) ** 2)
                if verbose > 0:
                    print(f"mu_pred:{mu_pred}")
                    print(f"std_pred:{std_pred}")

            elif p["loss"] == "mse":
                # see https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
                # calculate weights and prediction per model in ensemble
                weights = []
                pred = []
                for model_key, model in ensemble.items():
                    count = self.models_counts[ensemble_key][model_key]
                    weights.append(count / total_number_of_models)
                    pred.append(model.predict(x).reshape(-1, 1))
                    # pred.append(model(x).numpy()) #ONLY works when eager mode is enabled: first model(x) returns a tf.tensor -> convert to numpy array (reshape not necessary)
                if verbose > 0:
                    print(f"weights:{weights}")
                    print(f"predictions:{pred}")
                # calculate weighted mean
                mu_pred = np.average(pred, axis=0, weights=weights)
                # calculate weighted std (ddof==0 version)
                std_pred = np.average(
                    [(x - mu_pred) ** 2 for x in pred], axis=0, weights=weights
                )
            else:
                raise NotImplementedError(
                    "Loss {} not implemented yet.".format(p["loss"])
                )

            if p["normalize_data"]:
                print("Prediction function: Inverse-transform y(x-test)...")
                mu_pred = self.scaler_target.inverse_transform(mu_pred)
                std_pred = std_pred * self.scaler_target.scale_

            mu_pred = np.asarray(mu_pred, dtype=np.float32)
            std_pred = np.asarray(std_pred, dtype=np.float32)
            predictions[ensemble_key] = [mu_pred, std_pred]
        return predictions

    def calculate_mean_std(self, x: np.array) -> Dict[str, Tuple[np.array, np.array]]:

        """Calculates estimates of the model prediction and uncertainty for each model on a input point x,
        where a gaussian mixture model composed of áll individual nerual networks is assumed and
        model prediction is measured by taking the mean of that gaussian mixture, i.e., the mean of the mean ouptuts
        and uncertainty is measured by calculating the std of the gaussian mixture.

        Arguments
        ----------
        x :
            input data (features)

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_x]])
            {'Deep_Ensemble_1':(array([[mean_1],[mean_2]], dtype=float32),
                                    array([[sigma_1],[sigma_2]], dtype=float32)),
             'Deep_Ensemble_2':...
            }

        """

        predictions = self.predict(x)
        estimates = OrderedDict()
        for ensemble_key, v in predictions.items():
            estimates[ensemble_key] = v[0], v[1]
        return estimates

    def plot_histories(
        self,
        yscale: str = "linear",
        save_only: bool = False,
        absolutepath: Optional[str] = None,
    ) -> NoReturn:

        """Plots training histories per model.

        Arguments
        ----------
        yscale :
            y-axis scaling in plot.
        save_only :
            Bool if the plot should only be saved and not showed.
        absolutepath :
            Absolutepath including filename where the .pdf should be saved.

        """

        for ensemble_key, ensemble in self.models.items():
            plt.figure(figsize=(16, 9))
            for key, history in self.histories[ensemble_key].items():
                plt.plot(history["loss"], label=ensemble_key + "_" + key + ": loss")
                if history.get("val_loss", None) is not None:
                    plt.plot(history["val_loss"])
            plt.title("Training History", fontsize=20)
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(loc="best", prop={"size": 20})
            plt.grid()
            plt.yscale(yscale)

            if save_only:
                plt.savefig(
                    fname=absolutepath
                    + "_HDE{}.pdf".format(int(re.findall(r"\d+", ensemble_key)[0])),
                    format="pdf",
                    transparent=True,
                )
                plt.close()

    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""

        # Attributes
        self.parameters = OrderedDict()
        self.models = OrderedDict()
        self.models_counts = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []
        self.variable_hyperparameters_values = OrderedDict()
        self.current_seed_counter = None

    def save_models(self, absolutepath: str) -> NoReturn:

        """Saves models,models_counts, parameters, and histories of class instance.

        Arguments
        ----------
        absolutepath:
            Absolute path for saving.

        """

        # save tf models
        ensemble_number = 1
        for ensemble_key, ensemble in self.models.items():
            for model_key, model in ensemble.items():
                filename = "HDE_{}_".format(ensemble_number) + model_key
                model.save(os.path.join(absolutepath, filename) + ".h5")
            # save model counts in pickle file
            with open(
                os.path.join(absolutepath, "HDE_{}_counts.pkl".format(ensemble_number)),
                "wb",
            ) as f:
                pickle.dump(self.models_counts[ensemble_key], f)
            f.close()
            # save histories in pickle file
            with open(
                os.path.join(absolutepath, "HDE_{}_hist.pkl".format(ensemble_number)),
                "wb",
            ) as f:
                pickle.dump(self.histories[ensemble_key], f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(
                    absolutepath, "HDE_{}_parameters.pkl".format(ensemble_number)
                ),
                "wb",
            ) as f:
                pickle.dump(self.parameters[ensemble_key], f)
            f.close()
            # save variable hyperparameters in pickle file
            with open(
                os.path.join(
                    absolutepath,
                    "HDE_{}_variable_parameters.pkl".format(ensemble_number),
                ),
                "wb",
            ) as f:
                pickle.dump(self.variable_hyperparameters_values[ensemble_key], f)
            f.close()
            ensemble_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "HDE_all_parameters.txt"), "w") as f:
            for ensemble_key, p in self.parameters.items():
                f.write(ensemble_key + ":\n")
                for k, v in p.items():
                    f.write(k + ": " + str(v) + "\n")
                f.write("\n")
                f.write("Final Ensemble:\n")
                for k, v in self.models_counts[ensemble_key].items():
                    vp = ", ".join(
                        [
                            f"{k}={str(v)}"
                            for k, v in self.variable_hyperparameters_values[
                                ensemble_key
                            ][f"NN_1-{self._get_model_key_index(k)[1]}"].items()
                        ]
                    )
                    f.write(k + ": " + f"count={str(v)}" + ", " + vp + "\n")
                f.write("\n")
        f.close()
        print("\nModels saved in:", absolutepath)

    def load_models(
        self, absolutepath: str, model_numbers: Union[int, List[int]], verbose: int
    ) -> NoReturn:

        """Loads models, parameters, and histories for specified models via model_numbers
        and sets these values in the class instance HyperDeepEnsembles.

        Arguments
        ----------
        absolutepath:
            Absolute path for loading.
        model_numbers:
            Model numbers that should be loaded from the absolutepath location.
        verbose:
            Level of verbosity
        """

        # prepare
        self.reset_attributes()
        if not isinstance(model_numbers, list):
            model_numbers = [model_numbers]
        # filter model & parameter files
        ensemble_files = [
            [
                re.findall(r"HDE_" + str(model_number) + "_NN_\d{1,3}-\d{1,3}.h5", f)[0]
                for f in os.listdir(absolutepath)
                if len(
                    re.findall(
                        r"HDE_" + str(model_number) + "_NN_\d{1,3}-\d{1,3}.h5", f
                    )
                )
                > 0
            ]
            for model_number in model_numbers
        ]
        print(ensemble_files)
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "HDE_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        hist_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "HDE_{}_hist.pkl".format(model_number) for model_number in model_numbers
            ]
        ]
        variable_hyperparameters_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "HDE_{}_variable_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        models_counts_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "HDE_{}_counts.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        # load
        print("\nLoading the following models:")
        print(
            "**************************************************************************"
        )
        for (
            ensemble_file,
            parameter_file,
            hist_file,
            variable_hyperparameters_file,
            models_counts_file,
        ) in zip(
            ensemble_files,
            parameter_files,
            hist_files,
            variable_hyperparameters_files,
            models_counts_files,
        ):
            ensemble_key = "Hyper_Deep_Ensemble_{}".format(
                int(re.findall(r"\d+", parameter_file)[0])
            )  # here parameter file!
            print(ensemble_key)

            print("Loading file:", hist_file)
            with open(os.path.join(absolutepath, hist_file), "rb") as f:
                self.histories[ensemble_key] = pickle.load(f)
            f.close()
            print("Loading file:", parameter_file)
            with open(os.path.join(absolutepath, parameter_file), "rb") as f:
                self.parameters[ensemble_key] = pickle.load(f)
            f.close()
            print("Loading file:", variable_hyperparameters_file)
            with open(
                os.path.join(absolutepath, variable_hyperparameters_file), "rb"
            ) as f:
                self.variable_hyperparameters_values[ensemble_key] = pickle.load(f)
            f.close()
            print("Loading file:", models_counts_file)
            with open(os.path.join(absolutepath, models_counts_file), "rb") as f:
                self.models_counts[ensemble_key] = pickle.load(f)
            f.close()

            self.model_keys.append(ensemble_key)  # here needed!
            self.models[ensemble_key] = OrderedDict()  # here needed!
            for model_file in ensemble_file:
                print(model_file)
                if self.parameters[ensemble_key]["loss"] == "nll":
                    custom_softplus = softplus_wrapper(
                        self.parameters[ensemble_key]["softplus_min_var"]
                    )
                    model = load_model(
                        os.path.join(absolutepath, model_file),
                        compile=False,
                        custom_objects={
                            "gaussian_nll": gaussian_nll,
                            "custom_softplus": custom_softplus,
                        },
                    )
                elif self.parameters[ensemble_key]["loss"] == "mse":
                    model = load_model(
                        os.path.join(absolutepath, model_file), compile=False
                    )
                else:
                    raise NotImplementedError(
                        "{} loss is not implemented.".format(
                            self.parameters[ensemble_key]["loss"]
                        )
                    )
                self.models[ensemble_key][
                    re.search(r"NN_\d{1,3}-\d{1,3}", model_file)[0]
                ] = model
                if verbose > 0:
                    print("Summary:")
                    print(model.summary())
                    print()

    def _random_search(
        self, ensemble_key, s: float = 0.05, activation: str = "relu", verbose: int = 0
    ) -> NoReturn:
        # TODO: add description

        self._set_variable_hyperparameters_values(ensemble_key=ensemble_key)
        self._initialize_models(
            ensemble_key=ensemble_key, s=s, activation=activation, verbose=verbose
        )
        self._compile_models(ensemble_key=ensemble_key)

    def _hyper_ens(
        self,
        ensemble_key,
        score,
        score_single_model,
        verbose: int = 0,
        forbidden_fit_keys: Optional[list] = None,
        upper_bound_on_non_unique_models: int = 50,
    ) -> NoReturn:
        # TODO:add description

        # Fit models
        self._fit_models(
            ensemble_key,
            x=self.xtr,
            y=self.ytr,
            forbidden_fit_keys=forbidden_fit_keys,
            verbose=verbose,
        )

        full_ensemble = self.models[ensemble_key]
        parameters = self.parameters[ensemble_key]

        print(f"Build Greedy maxK-Ensemble from fitted {ensemble_key}:")
        print(
            "**************************************************************************"
        )
        # calculate best K-ensemble
        score_current_ensemble = np.inf
        current_ensemble = []
        current_ensemble_size = 0
        # loop over unique values
        while (
            len(set(current_ensemble)) < parameters["K"]
            and current_ensemble_size < upper_bound_on_non_unique_models
        ):
            if verbose > 0:
                print(f"\ncurrent ensemble size:{current_ensemble_size}")
                print(f"unique NNs:{len(set(current_ensemble))}")
                print(f"current ensemble:{[x[0] for x in current_ensemble]}")
                print(f"score current ensemble:{score_current_ensemble}")
            key_best_nn, score_ensemble = self._calculate_best_incremental_model(
                x=self.xtest,
                y=self.ytest,
                score=score,
                score_single_model=score_single_model,
                current_ensemble=current_ensemble,
                full_ensemble=full_ensemble,
                parameters=parameters,
                verbose=verbose,
            )
            if verbose > 0:
                print(
                    f"best:{key_best_nn} with", f"score of ensemble of:{score_ensemble}"
                )
            # we want at least two distinct models in our ensemble -> the 'or'
            if score_ensemble <= score_current_ensemble or len(current_ensemble) == 1:
                if len(current_ensemble) == 1 and verbose > 0:
                    print(
                        f"-> Forced adding of {key_best_nn}, since we want at least 2 NNs!"
                    )
                current_ensemble.append((key_best_nn, full_ensemble[key_best_nn]))
                current_ensemble_size += 1
                score_current_ensemble = score_ensemble
            else:
                print("NO NN IMPROVES THE SCORE -> BREAK CONDITION")
                break

        # print break condition
        a = len(set(current_ensemble)) < parameters["K"]
        b = current_ensemble_size < upper_bound_on_non_unique_models
        if a and b:
            pass
        elif a and not b:
            print(
                f"BOUND of {upper_bound_on_non_unique_models} ON NON UNIQUE NNs REACHED -> BREAK CONDITION"
            )
        elif not a and b:
            pp = parameters["K"]
            print(f"BOUND of {pp} ON UNIQUE NNs REACHED -> BREAK CONDITION")
        else:
            print("BOTH BOUNDS REACHED SIMULTANEOUSLY -> BREAK CONDITION")
        current_ensemble.sort()
        print(f"Return Ensemble:{[x[0] for x in current_ensemble]}")
        return current_ensemble

    def _stratify(
        self, ensemble_key, s: float = 0.05, activation: str = "relu", verbose: int = 0
    ) -> dict:
        # TODO: add description

        print(f"\nStratify: {list(self.models[ensemble_key].keys())}")
        print(
            "**************************************************************************"
        )

        p = self.parameters[ensemble_key]
        variable_hyperparameters_values = self.variable_hyperparameters_values[
            ensemble_key
        ]
        stratified_ensemble = OrderedDict()
        # row-wise stratification
        for i in range(1, p["K"] + 1):

            for model_key, model in self.models[ensemble_key].items():
                if i == 1:
                    # add model already found via random search
                    stratified_ensemble[model_key] = model
                    updated_seed = self.current_seed_counter[ensemble_key]
                else:
                    # variable hyper possible
                    l2reg = (
                        variable_hyperparameters_values[model_key]["l2reg"]
                        if isinstance(p["l2reg"], tuple)
                        else p["l2reg"]
                    )
                    dropout_prob = (
                        variable_hyperparameters_values[model_key]["dropout_prob"]
                        if isinstance(p["dropout_prob"], tuple)
                        else p["dropout_prob"]
                    )
                    # number of stratifications for a single model (K-1)*times
                    new_model_name = f"NN_{i}-{self._get_model_key_index(model_key)[1]}"
                    new_model, updated_seed = self._initialize_single_model(
                        modelname=new_model_name,
                        l2reg=l2reg,
                        dropout_prob=dropout_prob,
                        layers=p["layers"],
                        loss=p["loss"],
                        softplus_min_var=p["softplus_min_var"],
                        seed=self.current_seed_counter[ensemble_key],
                        s=s,
                        activation=activation,
                        verbose=verbose,
                    )
                    self._compile_single_model(model=new_model, parameters=p)
                    stratified_ensemble[new_model_name] = new_model
                    # update seed for every model in row if fixed_row_init=False
                    if not self.parameters[ensemble_key]["fixed_row_init"]:
                        self.current_seed_counter[ensemble_key] = updated_seed

            # update seed for after full row if fixed_row_init=True
            if self.parameters[ensemble_key]["fixed_row_init"]:
                self.current_seed_counter[ensemble_key] = updated_seed

        print(f"Return Ensemble:{list(stratified_ensemble.keys())}")
        return stratified_ensemble

    def _set_variable_hyperparameters_values(self, ensemble_key) -> NoReturn:
        # TODO: add description
        self.variable_hyperparameters_values[ensemble_key] = OrderedDict()
        for i in range(1, self.parameters[ensemble_key]["kappa"] + 1):
            modelname = f"NN_1-{i}"
            self.variable_hyperparameters_values[ensemble_key][
                modelname
            ] = OrderedDict()
            # dropout_prob
            dropout_prob = self.parameters[ensemble_key]["dropout_prob"]
            if isinstance(dropout_prob, tuple) and len(dropout_prob) == 2:
                u = np.random.uniform(
                    low=np.log10(dropout_prob[0]),
                    high=np.log10(dropout_prob[1]),
                    size=1,
                )[
                    0
                ]  # 'log' uniform
                self.variable_hyperparameters_values[ensemble_key][modelname][
                    "dropout_prob"
                ] = 10 ** (u)
            # l2reg
            l2reg = self.parameters[ensemble_key]["l2reg"]
            if isinstance(l2reg, tuple) and len(l2reg) == 2:
                u = np.random.uniform(
                    low=np.log10(l2reg[0]), high=np.log10(l2reg[1]), size=1
                )[
                    0
                ]  # 'log' uniform
                u_factor = (
                    1
                    - self.variable_hyperparameters_values[ensemble_key][modelname][
                        "dropout_prob"
                    ]
                ) / self.xtr.shape[0]
                self.variable_hyperparameters_values[ensemble_key][modelname][
                    "l2reg"
                ] = u_factor * 10 ** (u)

    def _initialize_models(
        self, ensemble_key, s: float = 0.05, activation: str = "relu", verbose: int = 0
    ) -> NoReturn:
        # TODO: update description
        """Initializes the neural network architectures.

        Arguments
        ----------
        s :
            Interval parameters for uniform random weight initialization in the intervall [-s,s]
        activation :
            Tf activation function.
        verbose :
            Verbosity level.

        """

        print(f"\nInitialize Hyper Deep Ensemble: {ensemble_key}")
        print(
            "**************************************************************************"
        )

        p = self.parameters[ensemble_key]
        ensemble = {}
        # default parameter
        layers = p["layers"]
        loss = p["loss"]
        softplus_min_var = p["softplus_min_var"]

        for i in range(1, p["kappa"] + 1):
            modelname = f"NN_1-{i}"
            # variable hyperparameters possible
            dropout_prob = (
                self.variable_hyperparameters_values[ensemble_key][modelname][
                    "dropout_prob"
                ]
                if isinstance(p["dropout_prob"], tuple)
                else p["dropout_prob"]
            )
            l2reg = (
                self.variable_hyperparameters_values[ensemble_key][modelname]["l2reg"]
                if isinstance(p["l2reg"], tuple)
                else p["l2reg"]
            )

            ensemble[modelname], updated_seed = self._initialize_single_model(
                modelname=modelname,
                l2reg=l2reg,
                dropout_prob=dropout_prob,
                layers=layers,
                loss=loss,
                softplus_min_var=softplus_min_var,
                seed=self.current_seed_counter[ensemble_key],
                verbose=verbose,
                s=s,
                activation=activation,
            )
            if verbose > 1:
                print(ensemble[modelname].summary())

            # update seed for every model in row if fixed_row_init=False
            if not self.parameters[ensemble_key]["fixed_row_init"]:
                self.current_seed_counter[ensemble_key] = updated_seed
        # update seed for after full row if fixed_row_init=True
        if self.parameters[ensemble_key]["fixed_row_init"]:
            self.current_seed_counter[ensemble_key] = updated_seed

        # store ensemble
        self.models[ensemble_key] = ensemble

    def _initialize_single_model(
        self,
        modelname,
        l2reg,
        dropout_prob,
        layers,
        loss,
        softplus_min_var,
        seed,
        s,
        activation,
        verbose,
    ) -> Model:
        # TODO: add description

        print("Initialize:", modelname)
        if verbose > 0:
            print(f"l2reg:{l2reg}")
            print(f"dropout_prob:{dropout_prob}")
            print(f"layers:{layers}")
            print(f"loss:{loss}")
            print(f"softplus_min_var:{softplus_min_var}")
            print(f"seed_init:{seed}")
            print()

        # input layer
        x_input = Input(shape=(layers[0],), name=modelname + "_input_layer")
        # first hidden layer
        l = Dense(
            layers[1],
            activation=activation,
            name=modelname + "_hidden_layer_{}".format(1),
            kernel_initializer=RandomUniform(
                minval=-s, maxval=s, seed=update_seed(seed, 0)
            ),
            bias_initializer=RandomUniform(
                minval=-s, maxval=s, seed=update_seed(seed, 1)
            ),
            kernel_regularizer=l2(l2reg),
            bias_regularizer=l2(l2reg),
        )(x_input)
        if dropout_prob != 0:
            l = Dropout(dropout_prob)(l)
        # hidden layers
        for i, n in enumerate(layers[2:-1]):
            l = Dense(
                n,
                activation=activation,
                name=modelname + "_hidden_layer_{}".format(i + 2),
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 3)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(l)
            if dropout_prob != 0:
                l = Dropout(dropout_prob)(l)
        if loss == "nll":
            # output layer with two parameters for each output dimension in case loss==nll:
            mu_output = Dense(
                layers[-1],
                activation="linear",
                name=modelname + "_output_layer_mu",
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 1) + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 1) + 3)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(l)
            sigma_output = Dense(
                layers[-1],
                activation=softplus_wrapper(min_var=softplus_min_var),
                name=modelname + "_output_layer_sigma",
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 2) + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 2) + 3)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(l)
            x_output = concatenate([mu_output, sigma_output])
        elif loss == "mse":
            # output layer with two parameters for each output dimension in case loss==nll:
            mu_output = Dense(
                layers[-1],
                activation="linear",
                name=modelname + "_output_layer_mu",
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 1) + 2)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 2 * (i + 1) + 3)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(l)
            x_output = mu_output
        else:
            raise NotImplementedError(
                "Loss {} is not implemented yet for deep ensembles.".format(loss)
            )

        seed = update_seed(
            seed, 2 * (i + 2) + 3 + 1
        )  # update seed for each model in ensemble to achieve diversity
        model = Model(inputs=[x_input], outputs=x_output)
        return (model, seed)

    def _compile_models(
        self,
        ensemble_key,
    ) -> NoReturn:

        """Compiles the neural network architectures.

        Arguments
        ----------
        ensemble_key:
            Key that determines which Hyper Deep Ensemble to compile.

        """

        print(f"\nCompile Hyper Deep Ensemble: {ensemble_key}")
        print(
            "**************************************************************************"
        )
        for model_key, model in self.models[ensemble_key].items():
            print("Compile:", model_key)
            self._compile_single_model(
                model=model, parameters=self.parameters[ensemble_key]
            )

    def _compile_single_model(self, model, parameters) -> NoReturn:
        # TODO: add description

        p = parameters
        if p["loss"] == "nll":
            loss_HDE = gaussian_nll
        elif p["loss"] == "mse":
            loss_HDE = "mse"
        else:
            raise NotImplementedError("{} loss is not implemented.".format(p["loss"]))
        # tensorflow version 2.1
        if p["optimizer"] == "Adam":
            optimizer = Adam(
                learning_rate=p["learning_rate"],
                beta_1=p["beta_1"],
                beta_2=p["beta_2"],
                epsilon=p["epsilon"],
                amsgrad=p["amsgrad"],
                name=p["optimizer"],
                clipnorm=p["clipnorm"],
            )
        elif p["optimizer"] == "SGD":
            optimizer = SGD(
                learning_rate=p["learning_rate"],
                momentum=p["momentum"],
                nesterov=p["nesterov"],
                name=p["optimizer"],
                clipnorm=p["clipnorm"],
            )
        else:
            raise NotImplementedError(
                "{} optimizer is not implemented.".format(p["optimizer"])
            )
        model.compile(
            optimizer=optimizer, loss=loss_HDE, experimental_run_tf_function=False
        )

    def _fit_models(
        self,
        ensemble_key,
        x: np.array,
        y: np.array,
        forbidden_fit_keys: Optional[List[str]] = [],
        verbose: int = 0,
    ) -> NoReturn:
        # TODO: update description
        """Fits the neural network architectures to specified data.

        Arguments
        ----------
        x :
            input data (features)
        y :
            output data (targets).
        verbose :
            Level of verbosity.
        """

        print(f"\nFit the Hyper Deep Ensemble: {ensemble_key}")
        print(
            "**************************************************************************"
        )
        if len(forbidden_fit_keys) > 0:
            print(f"Not fitting:{forbidden_fit_keys}")
        p = self.parameters[ensemble_key]
        start = datetime.now()
        history = OrderedDict()
        for model_key, model in self.models[ensemble_key].items():
            if model_key not in forbidden_fit_keys:
                print("Fit {}".format(model_key))
                tmp = model.fit(
                    x,
                    y,
                    epochs=p["epochs"],
                    batch_size=p["batch_size"],
                    verbose=0,
                    callbacks=[PredictionHistory_DE(x, y)],
                )
                history[model_key] = tmp.history
                # update history TODO:check
                if self.histories[ensemble_key].get(model_key) is None:
                    self.histories[ensemble_key][model_key] = history[model_key]
                else:
                    self.histories[ensemble_key][model_key] = {
                        loss_key: loss_value + history[model_key][loss_key]
                        for loss_key, loss_value in self.histories[ensemble_key][
                            model_key
                        ].items()
                    }
        end = datetime.now()
        diff = end - start
        print(
            "Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
            "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
        )
        self.parameters[ensemble_key]["actual_epochs"] = len(
            self.histories[ensemble_key][list(self.histories[ensemble_key].keys())[0]][
                "loss"
            ]
        )
        print()

    def _calculate_best_incremental_model(
        self,
        x,
        y,
        score,
        score_single_model,
        current_ensemble,
        full_ensemble,
        parameters,
        verbose=0,
    ) -> Tuple[str, float]:
        # TODO: add description

        best_score = np.inf
        best_key = None
        for model_key, model in full_ensemble.items():
            if verbose > 1:
                print(model_key)
                # print(f'current_ensemble:{current_ensemble + [(model_key,model)]}') already printed somewhere else
            tmp_score = self._calculate_ensemble_score(
                x=x,
                y=y,
                score=score,
                score_single_model=score_single_model,
                ensemble=current_ensemble + [(model_key, model)],
                parameters=parameters,
                verbose=verbose,
            )
            if tmp_score < best_score:
                best_score = tmp_score
                best_key = model_key
            if verbose > 1:
                print(f"tmp_score:{tmp_score}")
                print(f"best_score:{best_score}")
                print(f"best_key:{best_key}")
        return (best_key, best_score)

    def _calculate_ensemble_score(
        self, x, y, score, score_single_model, ensemble, parameters, verbose=0
    ) -> float:
        # TODO: add description

        if len(ensemble) == 0:
            raise ValueError("Input parameter ensemble is empty!")

        p = parameters
        if p["loss"] == "nll":
            sum_mu = np.zeros((len(x), 1))
            sum_mu_squared = np.zeros((len(x), 1))
            sum_var = np.zeros((len(x), 1))
            number_of_networks = 0
            for key_model_tuple in ensemble:
                _, model = key_model_tuple
                tmp = model.predict(x)[:, 0].reshape(-1, 1)
                sum_mu += tmp
                sum_mu_squared += (tmp) ** 2
                sum_var += model.predict(x)[:, 1].reshape(-1, 1)
                number_of_networks += 1
            mu_pred = sum_mu / number_of_networks
            std_pred = np.sqrt(
                (sum_var + sum_mu_squared) / number_of_networks
                - (sum_mu / number_of_networks) ** 2
            )
        elif p["loss"] == "mse":
            pred = [
                key_model_tuple[1].predict(x).reshape(-1, 1)
                for key_model_tuple in ensemble
            ]  # WHEN eager is disabled
            # pred = [key_model_tuple[1](x).numpy() for key_model_tuple in ensemble] # ONLY works if eager mode is enabled: first model(x) returns a tf.tensor -> convert to numpy array (reshape not necessary)
            mu_pred = np.mean(pred, axis=0)
            std_pred = np.std(pred, axis=0)
        else:
            raise NotImplementedError("Loss {} not implemented yet.".format(p["loss"]))
        mu_pred = np.asarray(mu_pred, dtype=np.float32)
        std_pred = np.asarray(std_pred, dtype=np.float32)
        if verbose > 2:
            print(f"mu_pred:{mu_pred}")
            print(f"std_pred:{std_pred}")
        # calc score for ensemble of size 1
        if len(ensemble) == 1:
            return score_single_model(y, mu_pred)
        # calc score for ensemble of size > 1
        else:
            return score(y, mu_pred, std_pred)

    def _get_model_key_index(self, model_key) -> Tuple[int, int]:
        # TODO: add description

        row, col = (
            int(re.findall(r"\d+", model_key.split("-")[0])[0]),
            int(re.findall(r"\d+", model_key.split("-")[1])[0]),
        )
        return (row, col)

    def _print_helper(self, ensemble_key: str) -> NoReturn:
        # TODO: add description

        print(
            "\n##########################################################################"
        )
        tmp_str = "["
        for k, v in self.models_counts[ensemble_key].items():
            tmp_str += f"({k},{v}), "
        tmp_str = tmp_str[:-2]
        tmp_str += "]"
        print(f"FINAL HYPER DEEP ENSEMBLE: {tmp_str}")
        print(
            "--------------------------------------------------------------------------"
        )
        tmp_str = ""
        for k, v in self.variable_hyperparameters_values[ensemble_key].items():
            tmp_str += f"{k}: "
            for k1, v1 in v.items():
                tmp_str += f"{k1}={v1} "
            tmp_str += "\n"
        tmp_str = tmp_str[:-2]
        print(tmp_str)
        print(
            "##########################################################################\n"
        )

    def _update_attributes(self, ensemble_key: str, valid_keys: list) -> NoReturn:
        # TODO: add description

        self.histories[ensemble_key] = OrderedDict(
            [(k, v) for k, v in self.histories[ensemble_key].items() if k in valid_keys]
        )
        # transform valid keys, since in variable_hyperparameters_values only keys with NN_1-i occur, i.e,
        # if NN_4-3 is selected as final model, we want the variable hyperparameters of NN_1-3.
        valid_keys = [f"NN_1-{self._get_model_key_index(k)[1]}" for k in valid_keys]
        self.variable_hyperparameters_values[ensemble_key] = OrderedDict(
            [
                (k, v)
                for k, v in self.variable_hyperparameters_values[ensemble_key].items()
                if k in valid_keys
            ]
        )

    def _calculate_dict_of_duplicates(self, key_value_list) -> dict:
        # TODO: add description

        return OrderedDict(Counter([x for (x, y) in key_value_list]))
