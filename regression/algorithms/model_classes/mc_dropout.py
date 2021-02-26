# -*- coding: utf-8 -*-
"""
This file contains the model class McDropout

"""
# Libs
from collections import OrderedDict
from itertools import product
import os
from datetime import datetime
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import RandomUniform
import pickle
import re
from typing import NoReturn, Union, List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s, update_seed

# %% Class for MC Dropout Approach


class McDropout:

    """
    MC Dropout (DO).

    ...

    Attributes
    ----------
    models : OrderedDict
        Tf.keras models.
    parameters : OrderedDict
        Parametersdict for each model.
    histories : OrderedDict
        Training histories for each model.

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

        """Constructor of the class McDropout."""

        # Attributes
        self.parameters = OrderedDict()
        self.models = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []

    def set_parameters(
        self,
        layers: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        epochs: Union[int, List[int]],
        batch_size: Union[int, List[int]],
        l2reg: Union[float, List[float]],
        optimizer_name: Union[str, List[str]],
        seed_init: Union[int, List[int]],
        loss: Union[str, List[str]],
        dropout_prob: Union[float, List[float]],
        optimizer_learning_rate: Optional[List[float]] = None,
        optimizer_clipnorm: Optional[List[float]] = None,
        optimizer_momentum: Optional[List[float]] = None,
        optimizer_nesterov: Optional[List[bool]] = None,
        optimizer_beta_1: Optional[List[float]] = None,
        optimizer_beta_2: Optional[List[float]] = None,
        optimizer_epsilon: Optional[List[float]] = None,
        optimizer_amsgrad: Optional[List[bool]] = None,
    ) -> NoReturn:

        """Sets the attributes of the class McDropout.

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
            Loss function for training the neural network.
        dropout_prob: float
            Probability of dropout (rate)
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
            "optimizer",
            "learning_rate",
            "momentum",
            "nesterov",
            "beta_1",
            "beta_2",
            "epsilon",
            "amsgrad",
            "clipnorm",
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
                optimizer_name,
                optimizer_learning_rate,
                optimizer_momentum,
                optimizer_nesterov,
                optimizer_beta_1,
                optimizer_beta_2,
                optimizer_epsilon,
                optimizer_amsgrad,
                optimizer_clipnorm,
            )
        )

        parameters = [OrderedDict(zip(parameter_keys, x)) for x in parameters_values]
        self.model_keys = ["MCDO_{}".format(i + 1) for i in range(len(parameters))]

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

        print("\nInitialize the following MC Dropout Models:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for key in self.models.keys():
                print(key)
                pretty_print_dict(self.parameters[key])
                print()
            print(
                "**************************************************************************"
            )
        for key in self.models.keys():
            print(key)
            p = self.parameters[key]

            layers = p["layers"]
            l2reg = p["l2reg"]
            seed = p["seed_init"]
            dropout = p["dropout_prob"]

            # input layer
            x_input = Input(shape=(layers[0],), name="input_layer")

            # first hidden layer
            l = Dense(
                layers[1],
                activation=activation,
                name="hidden_layer_{}".format(1),
                kernel_initializer=RandomUniform(minval=-s, maxval=s, seed=seed),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 1)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(x_input)
            if dropout != 0:
                l = Dropout(dropout)(l, training=True)
            # hidden layers
            for i, n in enumerate(layers[2:-1]):
                l = Dense(
                    n,
                    activation=activation,
                    name="hidden_layer_{}".format(i + 2),
                    kernel_initializer=RandomUniform(
                        minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 2)
                    ),
                    bias_initializer=RandomUniform(
                        minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 3)
                    ),
                    kernel_regularizer=l2(l2reg),
                    bias_regularizer=l2(l2reg),
                )(l)
                if dropout != 0:
                    l = Dropout(dropout)(l, training=True)
            # output layer
            x_output = Dense(
                layers[-1],
                activation="linear",
                name="output_layer",
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, -1)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, -2)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(l)
            self.models[key] = Model(inputs=[x_input], outputs=x_output)
            if verbose > 0:
                print("Architecture\n")
                print(self.models[key].summary())
        print()

    def compile_models(self, verbose: int = 0) -> NoReturn:

        """Compiles the neural network architectures.

        Arguments
        ----------
        verbose:
            Verbosity level.

        """

        print("\nCompile the following MC Dropout Models:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for key in self.models.keys():
                print(key)
                pretty_print_dict(self.parameters[key])
                print()
            print(
                "**************************************************************************"
            )
        for key in self.models.keys():
            print(key)
            p = self.parameters[key]
            model = self.models[key]
            # compile model
            # tensorflow version 2.1
            # (i) use adam
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
            if p["optimizer"] == "SGD":
                # (ii) plain vanilla gd
                optimizer = SGD(
                    learning_rate=p["learning_rate"],
                    momentum=p["momentum"],
                    nesterov=p["nesterov"],
                    name=p["optimizer"],
                    clipnorm=p["clipnorm"],
                )
            model.compile(
                optimizer=optimizer, loss=p["loss"], experimental_run_tf_function=False
            )

            self.models[key] = model
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

        print("\nFit the following MC Dropout Models:")
        print(
            "**************************************************************************"
        )
        if verbose > 0:
            for key in self.models.keys():
                print(key)
                pretty_print_dict(self.parameters[key])
                print()
            print(
                "**************************************************************************"
            )
        for key in self.models.keys():
            print(key)
            p = self.parameters[key]
            model = self.models[key]
            # fit model
            start = datetime.now()
            history = model.fit(
                x, y, epochs=p["epochs"], batch_size=p["batch_size"], verbose=verbose
            )
            end = datetime.now()
            diff = end - start
            print(
                "Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
            )
            self.models[key] = model
            if self.histories[key] is None:
                self.histories[key] = history.history
            else:
                self.histories[key] = {
                    loss_key: loss_value + history.history[loss_key]
                    for loss_key, loss_value in self.histories[key].items()
                }
            self.parameters[key]["actual_epochs"] = len(self.histories[key]["loss"])
        print()

    def predict(self, x: np.array, sample_size: int = 100) -> Dict[str, List[np.array]]:

        """Predicts the output for each model on a input point x.

        Arguments
        ----------
        x :
            input data (features).
        sample_size :
            number of predictions per input point x.

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model,
            e.g., for x = np.array([[x_1],[x_2]]) and sample_size=3 returns
            {'MCDO_1':[array([[y^1_1],[y^1_2]], dtype=float32),
                                    array([[y^2_1],[y^2_2]], dtype=float32),
                                    array([[y^3_1],[y^3_2]], dtype=float32)
             'MCDO_2':...
            }

        """

        predictions = OrderedDict()
        for key, model in self.models.items():
            predictions_list = []
            for i in range(sample_size):
                predictions_list.append(model.predict(x))
            predictions[key] = predictions_list
        return predictions

    def calculate_mean_std(
        self,
        x: np.array,
        sample_size: int = 100,
        predictions: Optional[np.array] = None,
    ) -> Dict[str, Tuple[np.array, np.array]]:

        """Calculates estimates of the model prediction and uncertainty for each model on a input point x,
        where model prediction is measured in terms of the mean over the sample_size many outputs of the neural network
        and uncertainty in terms of the std over sample_size many the outputs of the neural network

        Arguments
        ----------
        x :
            input data (features).
        sample_size :
            number of predictions per input point x.
        prediction :
            precalculated predictions from the model via predict()

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_x]])
            {'MCDO_1':(array([[mean_1],[mean_2]], dtype=float32),
                                    array([[std_1],[std_2]], dtype=float32)),
             'MCDO_2':...
            }

        """

        estimates = OrderedDict()
        if predictions is None:
            predictions = self.predict(x, sample_size=sample_size)
        for key, model in self.models.items():
            std_pred = np.std(predictions[key], axis=0)
            mu_pred = np.mean(predictions[key], axis=0)
            estimates[key] = mu_pred, std_pred
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

        plt.figure(figsize=(16, 9))
        for key, history in self.histories.items():
            plt.plot(history["loss"], label="MCDO " + key[-1] + ": loss")
            if history.get("val_loss", None) is not None:
                plt.plot(history["val_loss"])
        plt.title("Training History", fontsize=20)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(loc="best", prop={"size": 20})
        plt.grid()
        plt.yscale(yscale)

        if save_only:
            plt.savefig(fname=absolutepath + "_DO.pdf", format="pdf", transparent=True)
            plt.close()

    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""

        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.histories = OrderedDict()

    def save_models(self, absolutepath: str) -> NoReturn:

        """Saves models, parameters, and histories of class instance McDropout.

        Arguments
        ----------
        absolutepath:
            Absolute path for saving.

        """

        model_number = 1
        for key, model in self.models.items():
            # save tf models
            filename = "DO_{}".format(model_number)
            model.save(os.path.join(absolutepath, filename) + ".h5")
            # save histories in pickle file
            with open(
                os.path.join(absolutepath, "DO_{}_hist.pkl".format(model_number)), "wb"
            ) as f:
                pickle.dump(self.histories[key], f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(absolutepath, "DO_{}_parameters.pkl".format(model_number)),
                "wb",
            ) as f:
                pickle.dump(self.parameters[key], f)
            f.close()
            model_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "DO_all_parameters.txt"), "w") as f:
            for key, p in self.parameters.items():
                f.write(key + ":\n")
                for k, v in p.items():
                    f.write(k + ": " + str(v) + "\n")
                f.write("\n")
        f.close()
        print("\nModels saved in:", absolutepath)

    def load_models(
        self, absolutepath: str, model_numbers: Union[int, List[int]], verbose: int
    ) -> NoReturn:

        """Loads models, parameters, and histories for specified models via model_numbers
        and sets these values in the class instance Mc Dropout.

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
        model_files = [
            f
            for f in os.listdir(absolutepath)
            if f in ["DO_{}.h5".format(model_number) for model_number in model_numbers]
        ]
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "DO_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        hist_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in ["DO_{}_hist.pkl".format(model_number) for model_number in model_numbers]
        ]
        # load
        print("\nLoading the following models:")
        print(
            "**************************************************************************"
        )
        for model_file, parameter_file, hist_file in zip(
            model_files, parameter_files, hist_files
        ):
            key = "MCDO_{}".format(int(re.findall(r"\d+", model_file)[0]))
            print(key)
            print("Loading file:", hist_file)
            with open(os.path.join(absolutepath, hist_file), "rb") as f:
                self.histories[key] = pickle.load(f)
            f.close()
            print("Loading file:", parameter_file)
            with open(os.path.join(absolutepath, parameter_file), "rb") as f:
                self.parameters[key] = pickle.load(f)
            f.close()
            print("Loading file:", model_file)
            model = load_model(os.path.join(absolutepath, model_file), compile=False)
            self.models[key] = model
            if verbose > 0:
                print("\nSummary:")
                print(model.summary())
            print()
