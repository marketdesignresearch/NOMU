# -*- coding: utf-8 -*-
"""

This file contains the model class DeepEnsembles

"""

import os
import pickle
import re

# Libs
from collections import OrderedDict
from datetime import datetime
from itertools import product
from typing import Dict, List, NoReturn, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.initializers import RandomUniform  # , GlorotUniform
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s, update_seed
from algorithms.callbacks import PredictionHistory_DE
from algorithms.custom_activation_functions import softplus_wrapper
from algorithms.losses import gaussian_nll

# %% Class for Deep Ensembles Approach


class DeepEnsemble:

    """
    Deep Ensembles (DE).

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

        """Constructor of the class DeepEnsemble."""

        # Attributes
        self.parameters = OrderedDict()
        self.models = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []
        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()

    def set_parameters(
        self,
        layers: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        epochs: Union[int, List[int]],
        batch_size: Union[int, List[int]],
        l2reg: Union[float, List[float]],
        optimizer_name: Union[str, List[str]],
        seed_init: Union[int, List[int]],
        loss: Union[str, List[str]],
        number_of_networks: Union[int, List[int]],
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

        """Sets the attributes of the class DeepEnsemble.

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
            "number_of_networks",
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
        if not isinstance(number_of_networks, list):
            number_of_networks = [number_of_networks]
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
                number_of_networks,
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
            "Deep_Ensemble_{}".format(i + 1) for i in range(len(parameters))
        ]
        # Set Attributes
        i = 0
        for key in self.model_keys:
            self.parameters[key] = parameters[i]
            self.models[key] = None
            self.histories[key] = None
            i += 1

    def initialize_models(
        self, s: float = 0.05, activation: str = "relu", verbose: int = 0
    ) -> NoReturn:

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

        print("\nInitialize the following Deep Ensembles:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for ensemble_key in self.models.keys():
                print(ensemble_key)
                pretty_print_dict(self.parameters[ensemble_key])
                print()
            print(
                "**************************************************************************"
            )
        for ensemble_key in self.models.keys():
            print(ensemble_key)
            p = self.parameters[ensemble_key]
            layers = p["layers"]
            l2reg = p["l2reg"]
            number_of_networks = p["number_of_networks"]
            loss = p["loss"]
            softplus_min_var = p["softplus_min_var"]
            l2reg = p["l2reg"]
            seed = p["seed_init"]
            ensemble = {}
            for i in range(1, number_of_networks + 1):
                modelname = "NN_{}".format(i)
                print("Initialize:", modelname)
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
                # hidden layers
                for i, n in enumerate(layers[2:-1]):
                    l = Dense(
                        n,
                        activation=activation,
                        name=modelname + "_hidden_layer_{}".format(i + 2),
                        kernel_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * i + 2),
                        ),
                        bias_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * i + 3),
                        ),
                        kernel_regularizer=l2(l2reg),
                        bias_regularizer=l2(l2reg),
                    )(l)
                if loss == "nll":
                    # output layer with two parameters for each output dimension in case loss==nll:
                    mu_output = Dense(
                        layers[-1],
                        activation="linear",
                        name=modelname + "_output_layer_mu",
                        kernel_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 1) + 2),
                        ),
                        bias_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 1) + 3),
                        ),
                        kernel_regularizer=l2(l2reg),
                        bias_regularizer=l2(l2reg),
                    )(l)
                    # NOTE: THATS data noise output on quadratic scale i.e. sigma_n^2(x) (is then input to gaussian_nll)
                    sigma_output = Dense(
                        layers[-1],
                        activation=softplus_wrapper(min_var=softplus_min_var),
                        name=modelname + "_output_layer_sigma",
                        kernel_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 2) + 2),
                        ),
                        bias_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 2) + 3),
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
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 1) + 2),
                        ),
                        bias_initializer=RandomUniform(
                            minval=-s,
                            maxval=s,
                            seed=update_seed(seed, 2 * (i + 1) + 3),
                        ),
                        kernel_regularizer=l2(l2reg),
                        bias_regularizer=l2(l2reg),
                    )(l)
                    x_output = mu_output
                else:
                    raise NotImplementedError(
                        "Loss {} is not implemented yet for deep ensembles.".format(
                            loss
                        )
                    )
                ensemble[modelname] = Model(inputs=[x_input], outputs=x_output)
                if verbose > 0:
                    print(ensemble[modelname].summary())
                seed = update_seed(
                    seed, 2 * (i + 2) + 3 + 1
                )  # update seed for each model in ensemble to achieve diversity
            print()
            self.models[ensemble_key] = ensemble

    def compile_models(self, verbose: int = 0) -> NoReturn:

        """Compiles the neural network architectures.

        Arguments
        ----------
        verbose:
            Verbosity level.

        """

        print("\nCompile the following Deep Ensembles:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for ensemble_key in self.models.keys():
                print(ensemble_key)
                pretty_print_dict(self.parameters[ensemble_key])
                print()
            print(
                "**************************************************************************"
            )
        for ensemble_key in self.models.keys():
            print(ensemble_key)
            p = self.parameters[ensemble_key]
            for model_key, model in self.models[ensemble_key].items():
                print("Compile:", model_key)

                if p["loss"] == "nll":
                    loss_DE = gaussian_nll
                elif p["loss"] == "mse":
                    loss_DE = "mse"
                else:
                    raise NotImplementedError(
                        "{} loss is not implemented.".format(p["loss"])
                    )
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
                    optimizer=optimizer,
                    loss=loss_DE,
                    experimental_run_tf_function=False,
                )
        print()

    def fit_models(
        self,
        x: np.array,
        y: np.array,
        verbose: int = 0,
    ) -> NoReturn:

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

        print("\nFit the following Deep Ensembles:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for ensemble_key in self.models.keys():
                print(ensemble_key)
                pretty_print_dict(self.parameters[ensemble_key])
                print()
            print(
                "**************************************************************************"
            )
        for ensemble_key, ensemble in self.models.items():
            print(ensemble_key)
            p = self.parameters[ensemble_key]

            if p["normalize_data"]:
                print("Fit function: Fit & Transform x-train...")
                self.scaler_input.fit(x)
                x = self.scaler_input.transform(x)
                print("Fit function: Fit & Transform y-train...")
                y = np.array(y).reshape(-1, 1)
                self.scaler_target.fit(y)
                y = self.scaler_target.transform(y)

            start = datetime.now()
            history = OrderedDict()
            for model_key, model in ensemble.items():
                print("Fit {}".format(model_key))
                tmp = model.fit(
                    x,
                    y,
                    epochs=p["epochs"],
                    batch_size=p["batch_size"],
                    verbose=verbose,
                    callbacks=[PredictionHistory_DE(x, y)],
                )
                history[model_key] = tmp.history
            end = datetime.now()
            diff = end - start
            print(
                "Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
            )
            self.models[ensemble_key] = ensemble
            if self.histories[ensemble_key] is None:
                self.histories[ensemble_key] = history
            else:
                self.histories[ensemble_key] = {
                    model_key: {
                        loss_key: loss_value + history[model_key][loss_key]
                        for loss_key, loss_value in self.histories[ensemble_key][
                            model_key
                        ].items()
                    }
                    for model_key in self.models[ensemble_key].keys()
                }
            self.parameters[ensemble_key]["actual_epochs"] = len(
                self.histories[ensemble_key][
                    list(self.histories[ensemble_key].keys())[0]
                ]["loss"]
            )
            print()
        print()

    def predict(self, x: np.array) -> Dict[str, List[np.array]]:

        """Predicts the output for each model on a input point x.

        Arguments
        ----------
        x :
            input data (features).

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_2]])
            {'Deep_Ensemble_1':[array([[mean_1],[mean_2]], dtype=float32),
                                    array([[std_1],[std_2]], dtype=float32)],
             'Deep_Ensemble_2':...
            }

        """

        predictions = OrderedDict()
        for ensemble_key, ensemble in self.models.items():
            p = self.parameters[ensemble_key]

            if self.parameters[ensemble_key]["normalize_data"]:
                # print("Prediction function: Transform x-test...")
                if len(x.shape) == 1:  # if 1d, format to 2d array for transformation
                    x = x.reshape(-1, 1)
                x = self.scaler_input.transform(x)

            if p["loss"] == "nll":
                number_of_networks = p["number_of_networks"]
                sum_mu = np.zeros((len(x), 1))
                sum_mu_squared = np.zeros((len(x), 1))
                sum_var = np.zeros((len(x), 1))
                for model_key, model in ensemble.items():
                    tmp = model.predict(x)[:, 0].reshape(-1, 1)
                    sum_mu += tmp
                    sum_mu_squared += (tmp) ** 2
                    sum_var += model.predict(x)[:, 1].reshape(-1, 1)
                mu_pred = sum_mu / number_of_networks
                std_pred = np.sqrt(
                    (sum_var + sum_mu_squared) / number_of_networks
                    - (sum_mu / number_of_networks) ** 2
                )
            elif p["loss"] == "mse":
                pred = [
                    model.predict(x)[:, 0].reshape(-1, 1)
                    for model_key, model in ensemble.items()
                ]
                mu_pred = np.mean(pred, axis=0)
                std_pred = np.std(pred, axis=0)
            else:
                raise NotImplementedError(
                    "Loss {} not implemented yet.".format(p["loss"])
                )

            if self.parameters[ensemble_key]["normalize_data"]:
                # print("Prediction function: Inverse-transform y(x-test)...")
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
                    + "_DE{}.pdf".format(int(re.findall(r"\d+", ensemble_key)[0])),
                    format="pdf",
                    transparent=True,
                )
                plt.close()

    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""

        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.histories = OrderedDict()

    def save_models(self, absolutepath: str) -> NoReturn:

        """Saves models, parameters, and histories of class instance DeepEnsemble.

        Arguments
        ----------
        absolutepath:
            Absolute path for saving.

        """

        # save tf models
        ensemble_number = 1
        for ensemble_key, ensemble in self.models.items():
            for model_key, model in ensemble.items():
                filename = "DE_{}_".format(ensemble_number) + model_key
                model.save(os.path.join(absolutepath, filename) + ".h5")
            # save histories in pickle file
            with open(
                os.path.join(absolutepath, "DE_{}_hist.pkl".format(ensemble_number)),
                "wb",
            ) as f:
                pickle.dump(self.histories[ensemble_key], f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(
                    absolutepath, "DE_{}_parameters.pkl".format(ensemble_number)
                ),
                "wb",
            ) as f:
                pickle.dump(self.parameters[ensemble_key], f)
            f.close()
            ensemble_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "DE_all_parameters.txt"), "w") as f:
            for ensemble_key, p in self.parameters.items():
                f.write(ensemble_key + ":\n")
                for k, v in p.items():
                    f.write(k + ": " + str(v) + "\n")
                f.write("\n")
        f.close()
        print("\nModels saved in:", absolutepath)

    def load_models(
        self,
        absolutepath: str,
        model_numbers: Union[int, List[int]],
        verbose: int,
    ) -> NoReturn:

        """Loads models, parameters, and histories for specified models via model_numbers
        and sets these values in the class instance DeepEnsemble.

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
                re.findall(r"DE_" + str(model_number) + "_NN_\d{1,3}.h5", f)[0]
                for f in os.listdir(absolutepath)
                if len(re.findall(r"DE_" + str(model_number) + "_NN_\d{1,3}.h5", f)) > 0
            ]
            for model_number in model_numbers
        ]
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "DE_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        hist_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in ["DE_{}_hist.pkl".format(model_number) for model_number in model_numbers]
        ]
        # load
        print("\nLoading the following models:")
        print(
            "**************************************************************************"
        )
        for ensemble_file, parameter_file, hist_file in zip(
            ensemble_files, parameter_files, hist_files
        ):
            ensemble_key = "Deep_Ensemble_{}".format(
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
                self.models[ensemble_key][re.search(r"NN_\d", model_file)[0]] = model
                if verbose > 0:
                    print("Summary:")
                    print(model.summary())
                    print()
