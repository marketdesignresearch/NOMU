# -*- coding: utf-8 -*-
"""
This file contains the model class NOMU
"""


# Libs
import os
import pickle
import re
from collections import OrderedDict
from datetime import datetime
from itertools import product
from typing import Dict, List, NoReturn, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s, update_seed
from algorithms.callbacks import PredictionHistory, ReturnBestWeights
from algorithms.DataGenerator import DataGenerator
from algorithms.losses import r_loss_wrapper, squared_loss_wrapper
from algorithms.nll_calibration import nll_calibration
from algorithms.r_transform import r_transform_function

# %% Class for our NOMU-newtorks
class NOMU:

    """
    Uncertainty bounds for neural networks (NOMU).

    ...

    Attributes
    ----------
    models : OrderedDict
        Tf.keras models.
    parameters : OrderedDict
        Parametersdict for each model.
    flags : OrderedDict
        Tf.tensor for data dependent loss for each model.
    histories : OrderedDict
        Training histories for each model.
    r_transform_function: OrderedDict
        R_TRANSFORM instance in case r_transform is learned
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
    set_r_transform()
        Sets r_transform_function in case it applies
    set_augmentation_bounds()
        Sets bounds for augmented data (also for scaled versions if applicable)
    nll_calibration()
        Finds minimal NLPD (and corresponding c) over a grid of c_vaules.
    """

    def __init__(self) -> NoReturn:

        """Constructor of the class NOMU."""

        # Attributes
        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.flags = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []
        self.r_transform_function = OrderedDict()
        self.scaler_input = (
            StandardScaler()
        )  # Keeping track of normalization in Fit & Predict
        self.scaler_target = StandardScaler()
        self.x_min_aug = None  # Augmentation ranges (particularity of NOMU)
        self.x_max_aug = None  # See Fit func.
        self.x_min_aug_scaled = None  # Scaled augementation ranges
        self.x_max_aug_scaled = None  # See Fit func.

    def set_parameters(
        self,
        layers: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        epochs: Union[int, List[int]],
        batch_size: Union[int, List[int]],
        l2reg: Union[float, List[float]],
        optimizer_name: Union[str, List[str]],
        seed_init: Union[int, List[int]],
        MCaug: Union[bool, List[bool]],
        n_train: Union[int, List[int]],
        n_aug: Union[int, List[int]],
        side_layers: Union[Tuple[int, ...], List[Tuple[int, ...]]],
        mu_sqr: Union[float, List[float]],
        mu_exp: Union[float, List[float]],
        c_exp: Union[float, List[float]],
        r_transform: Union[str, List[str]],
        r_min: Union[float, List[float]],
        r_max: Union[float, List[float]],
        l2reg_sig: Optional[Union[float, List[float]]] = None,
        dropout_prob: Optional[Union[float, List[float]]] = None,
        RSN: Union[bool, List[bool]] = False,
        stable_aug_loss: Union[bool, List[bool]] = False,
        stable_loss: Union[int, List[int]] = 0,
        c_sqr_stable_aug_loss: Optional[List[float]] = None,
        c_negativ_stable: Optional[List[float]] = None,
        c_huber_stable: Optional[List[float]] = None,
        optimizer_learning_rate: Optional[List[float]] = None,
        optimizer_clipnorm: Optional[List[float]] = None,
        optimizer_momentum: Optional[List[float]] = None,
        optimizer_nesterov: Optional[List[bool]] = None,
        optimizer_beta_1: Optional[List[float]] = None,
        optimizer_beta_2: Optional[List[float]] = None,
        optimizer_epsilon: Optional[List[float]] = None,
        optimizer_amsgrad: Optional[List[bool]] = None,
        normalize_data: Optional[Union[bool, List[bool]]] = False,
        aug_in_training_range: Optional[Union[bool, List[bool]]] = False,
        aug_range_epsilon: Union[float, List[float]] = 0.05,
        connected_hidden: Optional[Union[bool, List[bool]]] = False,
    ) -> NoReturn:

        """Sets the attributes of the class NOMU.

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
            L2 weights (and bias) regularization for f architecture.
        optimizer_name: str
            Name of the optimizer for gradient descent updates.
        seed_init : int
            Seed for weight initialization.
        MCaug : bool
            Approximate integrals with MC and uniform sampling
        n_train: int
            Number of training points
        n_aug : int
            Number of augmented (artificial) points
        side_layers : list
            Layers of neural network architectures for side architecture "r-architecture".
        mu_sqr : float
            Weight parameter for squared loss term at training points
        mu_exp : float
            Weight parameter for exponential loss term at augmented points
        c_exp : float
            Parameter for exponential decay in in exponential loss term at augmented points
        r_transform :
            Prediction transform g() used for uncertainty ouput (=g(r)), either 'id' for g(x):=x or 'custom_min_max' for
            g(x):= (1-exp(-(relu(x) + r_min))*r_max
        r_min :
            Minimum uncertainty (=r) for numerical stability.
        r_max :
            Asymptotic maximum prior uncertainty (=r).
        l2reg_sig : int
            L2 weights (and bias) regularization for sigma architecture. If None, it is set to l2reg.
        dropout_prob:
            Dropout rate for the main architecture
        RSN : bool
            Random Shallow Neural Network (First hidden layer not trained)
        stable_aug_loss : bool
            Newer Stable loss version
        c_sqr_stable_aug_loss : float
            Parameter for stable_loss=1
        c_negativ_stable : float
            Parameter for stable_loss=2
        c_huber_stable : float
            Parameter for stable_loss=2
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
        aug_in_training_range:
            Boolean; if True, augmented data is sampled in trainings data range +-aug_range_epsilon%
        aug_range_epsilon:
            Percentage by which initial augmented data range is expanded (initial augmented data range is either training data range (if aug_in_training_range==True) or [-1, 1] (else))
        connected_hidden
        connections from the hidden layers of the main architecture to the hidden layers of the side-architecture
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

        # parameters for side architecture
        if l2reg_sig is None:
            l2reg_sig = l2reg

        # set parameters
        parameter_keys = [
            "layers",
            "epochs",
            "actual_epochs",
            "batch_size",
            "l2reg",
            "l2reg_sig",
            "dropout_prob",
            "seed_init",
            "MCaug",
            "n_train",
            "n_aug",
            "side_layers",
            "mu_sqr",
            "mu_exp",
            "c_exp",
            "r_transform",
            "r_min",
            "r_max",
            "RSN",
            "stable_aug_loss",
            "stable_loss",
            "c_sqr_stable_aug_loss",
            "c_negativ_stable",
            "c_huber_stable",
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
            "aug_in_training_range",
            "aug_range_epsilon",
            "connected_hidden",
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
        if not isinstance(l2reg_sig, list):
            l2reg_sig = [l2reg_sig]
        if not isinstance(dropout_prob, list):
            dropout_prob = [dropout_prob]
        if not isinstance(seed_init, list):
            seed_init = [seed_init]
        if not isinstance(MCaug, list):
            MCaug = [MCaug]
        if not isinstance(n_train, list):
            n_train = [n_train]
        if not isinstance(n_aug, list):
            n_aug = [n_aug]
        if not isinstance(side_layers, list):
            side_layers = [side_layers]
        if not isinstance(mu_sqr, list):
            mu_sqr = [mu_sqr]
        if not isinstance(mu_exp, list):
            mu_exp = [mu_exp]
        if not isinstance(c_exp, list):
            c_exp = [c_exp]
        if not isinstance(r_transform, list):
            r_transform = [r_transform]
        if not isinstance(r_min, list):
            r_min = [r_min]
        if not isinstance(r_max, list):
            r_max = [r_max]
        if not isinstance(RSN, list):
            RSN = [RSN]
        if not isinstance(stable_aug_loss, list):
            stable_aug_loss = [stable_aug_loss]
        if not isinstance(stable_loss, list):
            stable_loss = [stable_loss]
        if not isinstance(c_sqr_stable_aug_loss, list):
            c_sqr_stable_aug_loss = [c_sqr_stable_aug_loss]
        if not isinstance(c_negativ_stable, list):
            c_negativ_stable = [c_negativ_stable]
        if not isinstance(c_huber_stable, list):
            c_huber_stable = [c_huber_stable]
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
            optimizer_amsgrad = [optimizer_amsgrad]
        if not isinstance(normalize_data, list):
            normalize_data = [normalize_data]
        if not isinstance(aug_in_training_range, list):
            aug_in_training_range = [aug_in_training_range]
        if not isinstance(aug_range_epsilon, list):
            aug_range_epsilon = [aug_range_epsilon]
        if not isinstance(connected_hidden, list):
            connected_hidden = [connected_hidden]
        parameters_values = list(
            product(
                layers,
                epochs,
                actual_epochs,
                batch_size,
                l2reg,
                l2reg_sig,
                dropout_prob,
                seed_init,
                # loss,
                MCaug,
                n_train,
                n_aug,
                side_layers,
                mu_sqr,
                mu_exp,
                c_exp,
                r_transform,
                r_min,
                r_max,
                RSN,
                stable_aug_loss,
                stable_loss,
                c_sqr_stable_aug_loss,
                c_negativ_stable,
                c_huber_stable,
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
                aug_in_training_range,
                aug_range_epsilon,
                connected_hidden,
            )
        )

        parameters = [OrderedDict(zip(parameter_keys, x)) for x in parameters_values]
        self.model_keys = [
            "NOMU_Neural_Network_{}".format(i + 1) for i in range(len(parameters))
        ]

        # Set Attributes

        i = 0
        for key in self.model_keys:
            self.parameters[key] = parameters[i]
            self.models[key] = None
            self.flags[key] = None
            self.histories[key] = None
            # set r_transform function
            self.set_r_transform(
                r_transform_name=self.parameters[key]["r_transform"],
                model_key=key,
            )

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

        print("\nInitialize the following NOMU Models:")
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
            side_layers = p["side_layers"]
            l2reg = p["l2reg"]
            l2reg_sig = p["l2reg_sig"]
            seed = p["seed_init"]
            RSN = p["RSN"]
            dropout = p["dropout_prob"]
            # input layer
            x_input = Input(shape=(layers[0],), name="input_layer")
            flag = Input(shape=(1,), name="flag_input")
            # main architecture -------------------------------------------------------
            # first hidden layer
            y = Dense(
                layers[1],
                activation=activation,
                name="hidden_layer_{}".format(1),
                kernel_initializer=RandomUniform(minval=-s, maxval=s, seed=seed),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 1)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
                trainable=not (RSN),
            )(x_input)
            if dropout != 0 and dropout is not None:
                y = Dropout(dropout)(y)
            # hidden layers
            for i, n in enumerate(layers[2:-1]):
                y = Dense(
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
                    trainable=not (RSN),
                )(y)
                if dropout != 0 and dropout is not None:
                    y = Dropout(dropout)(y)
            # output layer
            y_output = Dense(
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
            )(y)

            # side architecture r -----------------------------------------------------
            # set new seed
            seed = update_seed(seed, 2 ** 6)
            # first hidden layer
            r = Dense(
                side_layers[1],
                activation=activation,
                name="r_hidden_layer_{}".format(1),
                kernel_initializer=RandomUniform(minval=-s, maxval=s, seed=seed),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 1)
                ),
                kernel_regularizer=l2(l2reg_sig),
                bias_regularizer=l2(l2reg_sig),
                trainable=not (RSN),
            )(x_input)
            # hidden layers
            for i, n in enumerate(side_layers[2:-1]):
                r = Dense(
                    n,
                    activation=activation,
                    name="r_hidden_layer_{}".format(i + 2),
                    kernel_initializer=RandomUniform(
                        minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 2)
                    ),
                    bias_initializer=RandomUniform(
                        minval=-s, maxval=s, seed=update_seed(seed, 2 * i + 3)
                    ),
                    kernel_regularizer=l2(l2reg_sig),
                    bias_regularizer=l2(l2reg_sig),
                    trainable=not (RSN),
                )(r)
            # concatenate last hidden of y and r
            y_r_concat = concatenate([y, r])

            r_output = Dense(
                side_layers[-1],
                activation="linear",
                name="r_output_layer",
                kernel_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, -1)
                ),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, -2)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
            )(y_r_concat)

            self.models[key] = Model(
                inputs=[x_input, flag], outputs=[y_output, r_output]
            )
            self.flags[key] = flag
        print()

    def compile_models(self, verbose: int = 0) -> NoReturn:

        """Compiles the neural network architectures.

        Arguments
        ----------
        verbose:
            Verbosity level.

        """

        print("\nCompile the following NOMU Models:")
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
            flag = self.flags[key]
            # define loss
            custom_loss = [
                squared_loss_wrapper(flag=flag, n_train=p["n_train"], n_aug=p["n_aug"]),
                r_loss_wrapper(
                    flag=flag,
                    mu_sqr=p["mu_sqr"],
                    mu_exp=p["mu_exp"],
                    c_exp=p["c_exp"],
                    n_train=p["n_train"],
                    n_aug=p["n_aug"],
                    stable_loss=p["stable_loss"],
                    c_2=p["c_sqr_stable_aug_loss"],
                    c_negativ_stable=p["c_negativ_stable"],
                    c_huber_stable=p["c_huber_stable"],
                ),
            ]
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
                    clipnorm=None,
                )
            model.compile(
                optimizer=optimizer,
                loss=custom_loss,
                experimental_run_tf_function=False,
            )
            self.models[key] = model
        print()

    def augmentation_bounds(
        self, x: np.array, aug_in_training_range=False, aug_range_epsilon=0.05
    ) -> List[np.array]:

        """Calculates the bounds of the intervals of the augmented data

        Arguments
        ----------
        x :
            input data (features)
        aug_in_training_range:
            Boolean; if True, augmented data is sampled in trainings data range +-aug_range_epsilon%
        aug_range_epsilon:
            Percentage by which initial augmented data range is expanded (initial augmented data range is either training data range (if aug_in_training_range==True) or [-1, 1] (else))


        """
        if aug_in_training_range:
            x_min = x.min(axis=0)
            x_max = x.max(axis=0)
        else:
            x_min = -1
            x_max = 1
        diff = x_max - x_min
        x_min = x_min - aug_range_epsilon * diff
        x_max = x_max + aug_range_epsilon * diff
        return (x_min, x_max)

    def set_augmentation_bounds(
        self,
        x: np.array,
        normalize_data=False,
        aug_in_training_range=False,
        aug_range_epsilon=0.05,
    ) -> NoReturn:

        """Sets the bounds of the intervals of the augmented data

        Arguments
        ----------
        x :
            input data (features)
        normalize_data:
            boolean if bounds based on normalized data should also be set
        aug_in_training_range:
            Boolean; if True, augmented data is sampled in trainings data range +-aug_range_epsilon%
        aug_range_epsilon:
            Percentage by which initial augmented data range is expanded (initial augmented data range is either training data range (if aug_in_training_range==True) or [-1, 1] (else))


        """
        self.x_min_aug, self.x_max_aug = self.augmentation_bounds(
            x,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )

        if normalize_data:
            self.scaler_input.fit(x)
            x = self.scaler_input.transform(x)

            (self.x_min_aug_scaled, self.x_max_aug_scaled,) = self.augmentation_bounds(
                x,
                aug_in_training_range=aug_in_training_range,
                aug_range_epsilon=aug_range_epsilon,
            )

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

        # fit NOMU models
        print("\nFit the following NOMU Models:")
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

            x_orig = x[: p["n_train"], :-1]
            x_aug = x[p["n_train"] :, :-1]

            # Set augmentation bounds
            self.set_augmentation_bounds(
                x_orig,
                normalize_data=p["normalize_data"],
                aug_in_training_range=p["aug_in_training_range"],
                aug_range_epsilon=p["aug_range_epsilon"],
            )
            x_min_aug = self.x_min_aug
            x_max_aug = self.x_max_aug

            if p["normalize_data"]:
                x_aug = self.scaler_input.transform(x_aug)
                x_orig = self.scaler_input.transform(x_orig)
                x_hat = np.concatenate((x_orig, x_aug), axis=0)
                x = np.concatenate((x_hat, x[:, -1].reshape(-1, 1)), axis=1)
                self.scaler_target.fit(y[: p["n_train"]])
                y = self.scaler_target.transform(y)

                x_min_aug = self.x_min_aug_scaled
                x_max_aug = self.x_max_aug_scaled

            # set up generator
            generator = DataGenerator(
                batch_size=p["batch_size"],
                x=x,
                y=y,
                n_train=p["n_train"],
                n_aug=p["n_aug"],
                MCaug=p["MCaug"],
                x_min_aug=x_min_aug,
                x_max_aug=x_max_aug,
                joint_loss=True,
            )
            # fit model
            best_weights_callback = ReturnBestWeights(
                monitor="loss", verbose=1, mode="min", baseline=None
            )
            start = datetime.now()
            history = model.fit(
                x=generator,
                epochs=p["epochs"],
                verbose=verbose,
                callbacks=[best_weights_callback, PredictionHistory(generator)],
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

    def predict(self, x: np.array, model_key: str = None) -> Dict[str, List[np.array]]:

        """Predicts the output for each model on a input point x.

        Arguments
        ----------
        x :
            input data (features).
        model_key:
            If not None, predictions are only calculated for the given model key.

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_2]])
            {'NOMU_Neural_Network_1':[array([[y_1],[y_2]], dtype=float32),
                                    array([[r_1],[r_2]], dtype=float32)],
             'NOMU_Neural_Network_2':...
            }

        """
        if not model_key:
            iterable = self.model_keys
        else:
            iterable = [model_key]

        predictions = OrderedDict()
        for key in iterable:
            model = self.models[key]
            r_transform = self.parameters[key]["r_transform"]

            xFlag = np.zeros((x.shape[0], 1))

            try:
                if self.parameters[key]["normalize_data"]:
                    x = np.array(x)  # B.c. of warning regarding pd & numpy
                    if len(x.shape) == 1:
                        x = x.reshape(-1, 1)

                    x = self.scaler_input.transform(x)
            except (KeyError):
                pass

            prediction = model.predict([x, xFlag])

            if r_transform in ["id", "custom_min_max", "relu_cut"]:
                prediction[1] = self.r_transform_function[key](prediction[1])
            elif r_transform[:7] == "learned":
                prediction[1] = (
                    self.r_transform_function[key].predict(prediction[1]).reshape(-1, 1)
                )

            else:
                raise NotImplementedError(
                    "r_transform {} not implemented.".format(r_transform)
                )

            try:
                if self.parameters[key]["normalize_data"]:
                    prediction = np.array(prediction)
                    prediction[0] = self.scaler_target.inverse_transform(prediction[0])
                    prediction[1] = prediction[1] * self.scaler_target.scale_
            except (KeyError):
                pass

            predictions[key] = prediction

        return predictions

    def evaluate(
        self,
        x: np.array,
        y: np.array,
    ) -> Dict[str, List[np.array]]:

        """Evaluates loss metrics for each model on a data sample x,y.

        Arguments
        ----------
        x :
            input data (features).
        y :
            output data (targets).

        Returns
        -------
        evaluations:
            A dictionary that stores the loss metrics for each model, e.g., for x = np.array([[x_1],[x_2]])
            {'NOMU_Neural_Network_1':[array([loss], dtype=float32),
                                    array([r_loss], dtype=float32),
                                    array([y_loss], dtype=float32)],
             'NOMU_Neural_Network_2':...
            }

        """

        evaluations = OrderedDict()
        for key, model in self.models.items():
            evaluations[key] = model.evaluate(x=x, y=y)
        return evaluations

    def calculate_mean_std(
        self, x: np.array, model_key: str = None
    ) -> Dict[str, Tuple[np.array, np.array]]:

        """Calculates estimates of the model prediction and uncertainty for each model on a input point x,
        where model prediction is measured in terms of the output of the y-architecture
        and uncertainty in terms of the r-architecture.

        Arguments
        ----------
        x :
            input data (features)
        r_transform :
            Applied transform g() to the raw r-ouput. Either 'id' for g(r)=r or 'custom_min_max' for
            g(x):= (1-exp(-(relu(x) + r_min))*r_max. If not specified self.r_transform is used as transform.
        r_min :
            Minimum uncertainty (=r) for numerical stability. If not specified self.r_min is used.
        r_max :
            Asymptotic maximum prior uncertainty (=r).  If not specified self.r_max is used.
        model_key:
            If not None, mean and std estimates are only calculated for the given model_key.

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_x]])
            {'NOMU_Neural_Network_1':(array([[y_1],[y_2]], dtype=float32),
                                    array([[r_1],[r_2]], dtype=float32)),
             'NOMU_Neural_Network_2':...
            }

        """
        if not model_key:
            iterable = self.model_keys
        else:
            iterable = [model_key]

        estimates = OrderedDict()
        for key in iterable:
            predictions = self.predict(x=x, model_key=key)
            std_pred = predictions[key][1]
            mu_pred = predictions[key][0]
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
            Absolutepath excluding filename where the .pdf should be saved.

        """

        plt.figure(figsize=(16, 9))
        for key, history in self.histories.items():
            plt1 = plt.plot(
                history.get("r_output_layer_loss", None),
                label=key + ": r_loss",
                linestyle="dotted",
            )
            plt.plot(
                history.get("output_layer_loss", None),
                label=key + ": y_loss",
                color=plt1[0].get_color(),
                linestyle="dashed",
            )
            plt.plot(
                history["loss"],
                label=key + ": loss",
                color=plt1[0].get_color(),
            )
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
                fname=absolutepath + "_NOMU.pdf", format="pdf", transparent=True
            )
            plt.close()

    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""

        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.flags = OrderedDict()
        self.histories = OrderedDict()
        self.model_keys = []

    def save_models(self, absolutepath: str) -> NoReturn:

        """Saves models, parameters, and histories of class instance NOMU.

        Arguments
        ----------
        absolutepath:
            Absolute path for saving.

        """

        model_number = 1
        for key, model in self.models.items():
            # save tf models
            filename = "NOMU_{}".format(model_number)
            model.save(os.path.join(absolutepath, filename) + ".h5")
            # save histories in pickle file
            with open(
                os.path.join(absolutepath, "NOMU_{}_hist.pkl".format(model_number)),
                "wb",
            ) as f:
                pickle.dump(self.histories[key], f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(
                    absolutepath, "NOMU_{}_parameters.pkl".format(model_number)
                ),
                "wb",
            ) as f:
                pickle.dump(self.parameters[key], f)
            f.close()
            model_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "NOMU_all_parameters.txt"), "w") as f:
            for key, p in self.parameters.items():
                f.write(key + ":\n")
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
        and sets these values in the class instance NOMU.

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
            if f
            in ["NOMU_{}.h5".format(model_number) for model_number in model_numbers]
        ]
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "NOMU_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        hist_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "NOMU_{}_hist.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        # load
        print("\nLoading the following models:")
        print(
            "**************************************************************************"
        )
        for model_file, parameter_file, hist_file in zip(
            model_files, parameter_files, hist_files
        ):
            key = "NOMU_Neural_Network_{}".format(
                int(re.findall(r"\d+", model_file)[0])
            )
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
            model_NOMU = load_model(
                os.path.join(absolutepath, model_file), compile=False
            )
            self.models[key] = model_NOMU
            self.flags[key] = model_NOMU.inputs[-1]
            self.model_keys.append(key)
            self.set_r_transform(
                r_transform_name=self.parameters[key]["r_transform"],
                r_min=self.parameters[key]["r_min"],
                r_max=self.parameters[key]["r_max"],
                model_key=key,
            )
            if verbose > 0:
                print("\nSummary:")
                print(model_NOMU.summary())
            print()

    def set_r_transform(
        self, r_transform_name, r_min=None, r_max=None, model_key=None
    ) -> NoReturn:
        if not model_key:
            iterable = self.model_keys
        else:
            iterable = [model_key]
        for key in iterable:
            # set or retrieve r_transform_name
            self.parameters[key]["r_transform"] = r_transform_name

            # if r_transform_name[:7] == "learned":
            #     r_trafo = R_TRANSFORM()
            #     r_trafo.load_transform(
            #         absolutepath=r_transform_name[8:], verbose=1
            #     )
            #     self.r_transform_function[key] = r_trafo
            if r_transform_name in ["custom_min_max", "relu_cut"]:
                # set or retrieve r_min, r_max
                if r_min is None:
                    r_min = self.parameters[key]["r_min"]
                else:
                    self.parameters[key]["r_min"] = r_min
                if r_max is None:
                    r_max = self.parameters[key]["r_max"]
                else:
                    self.parameters[key]["r_max"] = r_max

                self.r_transform_function[key] = lambda r: r_transform_function(
                    r=r,
                    r_min=r_min,
                    r_max=r_max,
                    r_transform=r_transform_name,
                    learned_R_TRANSFORM=None,
                )
            elif r_transform_name == "id":
                self.r_transform_function[key] = lambda r: r
            else:
                raise NotImplementedError(f"{r_transform_name} not implemented!\n")

    def nll_calibration_r_grids(
        self,
        r_min_grid,
        r_max_grid,
        c_grid,
        x_val,
        y_val,
        add_nlpd_constant=False,
    ):
        nll_dict_r_grids = {}
        for key in self.model_keys:
            r_transform_name = self.parameters[key]["r_transform"]
            if r_transform_name in ["custom_min_max", "relu_cut"]:
                nll_dict_r_grids.update(
                    {key: {"nll": np.inf, "c": None, "r_min": 0, "r_max": 0}}
                )
                for r_min, r_max in product(r_min_grid, r_max_grid):
                    self.set_r_transform(
                        r_transform_name=r_transform_name,
                        r_min=r_min,
                        r_max=r_max,
                        model_key=key,
                    )
                    nll_c = nll_calibration(
                        model=self,
                        x_val=x_val,
                        y_val=y_val,
                        c_grid=c_grid,
                        add_nlpd_constant=add_nlpd_constant,
                        model_key=key,
                    )

                    if nll_c[key]["nll_min"] < nll_dict_r_grids[key]["nll"]:
                        nll_dict_r_grids[key]["nll"] = nll_c[key]["nll_min"]
                        nll_dict_r_grids[key]["c"] = nll_c[key]["c_min"]
                        nll_dict_r_grids[key]["r_min"] = r_min
                        nll_dict_r_grids[key]["r_max"] = r_max
                    # set NOMU r_transform parameters to calibrated values
                self.set_r_transform(
                    r_transform_name=r_transform_name,
                    r_min=nll_dict_r_grids[key]["r_min"],
                    r_max=nll_dict_r_grids[key]["r_max"],
                    model_key=key,
                )
        return nll_dict_r_grids
