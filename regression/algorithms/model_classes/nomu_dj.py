# -*- coding: utf-8 -*-
"""
This file contains the model class NOMU disjoint

"""

# Libs
from collections import OrderedDict
import os
from datetime import datetime
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Dense, concatenate, Dropout
from tensorflow.keras.initializers import RandomUniform
import pickle
import re
from typing import NoReturn, Union, List, Optional
import matplotlib.pyplot as plt

# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s, update_seed
from algorithms.model_classes.nomu import NOMU
from algorithms.losses import r_loss_wrapper, sum_of_squared_loss
from algorithms.DataGenerator import DataGenerator
from algorithms.callbacks import PredictionHistory, ReturnBestWeights

# %% Class for our disjoint UB-networks
class NOMU_DJ(NOMU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_models = OrderedDict()
        self.sigma_models = OrderedDict()

    #%%
    def set_parameters(self, *args, **kwargs):
        super().set_parameters(*args, **kwargs)

        # Model attribute is split into mean and sigma and rename NOMU to NOMU_DJ
        newkeys = []
        for key in self.model_keys:
            newkey = "NOMU_DJ" + key[4:]
            self.parameters[newkey] = self.parameters.pop(key)
            self.flags[newkey] = self.flags.pop(key)
            self.histories[newkey] = self.histories.pop(key)
            self.models[newkey] = self.models.pop(key)

            self.mean_models[newkey] = None
            self.sigma_models[newkey] = None
            self.histories[newkey] = OrderedDict(
                {"mean_model": None, "sigma_model": None}
            )
            newkeys.append(newkey)
        self.model_keys = newkeys

    #%%

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
        # initialize main NOMU model
        super().initialize_models(s=s, activation=activation, verbose=verbose)

        print("\nInitialize the following NOMU_DJ Models:")
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

            self.mean_models[key] = Model(inputs=[x_input], outputs=[y_output])

            # side architecture r -----------------------------------------------------

            # PHANTOM MODEL FOR MEAN
            # main architecture phantom -----------------------------------------------
            # first hidden layer
            yp = Dense(
                layers[1],
                activation=activation,
                name="hidden_layer_{}".format(1),
                kernel_initializer=RandomUniform(minval=-s, maxval=s, seed=seed),
                bias_initializer=RandomUniform(
                    minval=-s, maxval=s, seed=update_seed(seed, 1)
                ),
                kernel_regularizer=l2(l2reg),
                bias_regularizer=l2(l2reg),
                trainable=False,
            )(x_input)
            # if dropout != 0 and dropout is not None: y = Dropout(dropout)(yp)
            # hidden phanotm layers
            for i, n in enumerate(layers[2:-1]):
                yp = Dense(
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
                    trainable=False,
                )(yp)
                # if dropout != 0 and dropout is not None: y = Dropout(dropout)(yp)

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

            # concatenate last hidden of phantom y and r
            y_r_concat = concatenate([yp, r])

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

            self.sigma_models[key] = Model(inputs=[x_input, flag], outputs=[r_output])
            self.flags[key] = flag

        # update all parameters of main architectures
        self.update_all_weights()

        print()

    #%%

    def compile_models(self, verbose: int = 0) -> NoReturn:

        """Compiles the neural network architectures.

        Arguments
        ----------
        verbose:
            Verbosity level.

        """

        print("\nCompile the following NOMU_DJ Models:")
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

        self.compile_mean_models()
        self.compile_sigma_models()

        # update all parameters of main architectures
        self.update_all_weights()

        print()

    #%%
    def compile_mean_models(self) -> NoReturn:

        """Compiles the mean network architectures."""

        print("\nCompiling Mean Models:")
        print(
            "**************************************************************************"
        )
        for key, mod in self.mean_models.items():
            print(key)
            p = self.parameters[key]

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
            mod.compile(
                optimizer=optimizer,
                loss=sum_of_squared_loss,
                experimental_run_tf_function=False,
            )
            self.mean_models[key] = mod

            print()

    #%%
    def compile_sigma_models(self) -> NoReturn:

        """Compiles the sigma network architectures."""

        print("\nCompiling Sigma Models:")
        print(
            "**************************************************************************"
        )
        for key, mod in self.sigma_models.items():
            print(key)
            p = self.parameters[key]
            flag = self.flags[key]

            # define loss
            custom_loss = r_loss_wrapper(
                flag=flag,
                mu_sqr=p["mu_sqr"],
                mu_exp=p["mu_exp"],
                c_exp=p["c_exp"],
                n_train=p["n_train"],
                n_aug=p["n_aug"],
                stable_aug_loss=p["stable_aug_loss"],
                c_2=p["c_sqr_stable_aug_loss"],
            )
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
            mod.compile(
                optimizer=optimizer,
                loss=custom_loss,
                experimental_run_tf_function=False,
            )
            self.sigma_models[key] = mod

            print()

    #%%
    def fit_models(
        self,
        x: np.array,
        y: np.array,
        verbose: int = 0,
        x_min_aug: float = -1 - 0.1,
        x_max_aug: float = 1 + 0.1,
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
        x_min_aug :
            Min of box bound for augmented data points.
        x_max_aug :
            Max of box bound for augmented data points.

        """

        # fit NOMU_dj models
        print("\nFit the following NOMU_DJ Models:")
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
        # fit mean models
        self.fit_mean_models(x=x, y=y, verbose=verbose)

        # set parameters of main architecture in sigma model to the ones of the mean model
        self.update_sigma_weights()

        # fit sigma model
        self.fit_sigma_models(
            x=x, y=y, verbose=verbose, x_min_aug=x_min_aug, x_max_aug=x_max_aug
        )

        # update all parameters of main architectures
        self.update_all_weights()
        print()

    # %%
    def update_sigma_weights(self) -> NoReturn:
        """Updates weights of sigma network."""

        for key in self.models.keys():
            print(f"\nUpdating main architecture of sigma for: {key}")
            print(
                "**************************************************************************"
            )

            for l in self.sigma_models[key].layers:
                if l._name[:6] == "hidden":
                    wm = self.mean_models[key].get_layer(l._name).get_weights()
                    l.set_weights(wm)

        print()

    # %%
    def update_all_weights(self) -> NoReturn:

        """Updates weights of sigma and thereafter NOMU network."""

        # update sigma model
        self.update_sigma_weights()

        # update NOMU model
        for key in self.models.keys():
            print(f"\nUpdating entire architecture of NOMU model for: {key}")
            print(
                "**************************************************************************"
            )
            # update hidden layers
            for l in self.models[key].layers:
                if "hidden" in l._name:
                    ws = self.sigma_models[key].get_layer(l._name).get_weights()
                    l.set_weights(ws)
            # update final output layer
            out = self.mean_models[key].get_layer("output_layer").get_weights()
            self.models[key].get_layer("output_layer").set_weights(out)
            # update final r-output layer
            r_out = self.sigma_models[key].get_layer("r_output_layer").get_weights()
            self.models[key].get_layer("r_output_layer").set_weights(r_out)

        print()

    # %%
    def fit_mean_models(self, x: np.array, y: np.array, verbose: int = 0) -> NoReturn:
        """Fits the mean architectures to specified data.

        Arguments
        ----------
        x :
            input data (features)
        y :
            output data (targets).
        verbose :
            Level of verbosity.
        """

        print("\nFit Mean Models:")
        print(
            "**************************************************************************"
        )
        for key, model in self.mean_models.items():
            print(key)
            p = self.parameters[key]
            n_train = p["n_train"]

            # fit mean model
            best_weights_callback = ReturnBestWeights(
                monitor="loss", verbose=1, mode="min", baseline=None
            )
            start = datetime.now()
            history = model.fit(
                x=x[:n_train, :-1],
                y=y[:n_train],
                epochs=p["epochs"],
                batch_size=p["batch_size"],
                verbose=verbose,
                callbacks=[best_weights_callback],
            )

            end = datetime.now()
            diff = end - start
            print(
                "Mean Model Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
            )

            self.mean_models[key] = model
            if self.histories[key]["mean_model"] is None:
                self.histories[key]["mean_model"] = history.history
            else:
                self.histories[key]["mean_model"] = {
                    loss_key: loss_value + history.history[loss_key]
                    for loss_key, loss_value in self.histories[key][
                        "mean_model"
                    ].items()
                }
            self.parameters[key]["actual_epochs_mean"] = len(
                self.histories[key]["mean_model"]["loss"]
            )

        print()

    # %%
    def fit_sigma_models(
        self,
        x: np.array,
        y: np.array,
        verbose: int = 0,
        x_min_aug: float = -1 - 0.1,
        x_max_aug: float = 1 + 0.1,
    ) -> NoReturn:
        """Fits the sigma architectures to specified data.

        Arguments
        ----------
        x :
            input data (features)
        y :
            output data (targets).
        verbose :
            Level of verbosity.
        x_min_aug :
            Min of box bound for augmented data points.
        x_max_aug :
            Max of box bound for augmented data points.

        """

        print("\nFit Sigma Models:")
        print(
            "**************************************************************************"
        )
        for key, model in self.sigma_models.items():
            print(key)
            p = self.parameters[key]

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
                joint_loss=False,
            )

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
                "Sigma Model Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
            )

            self.sigma_models[key] = model
            if self.histories[key]["sigma_model"] is None:
                self.histories[key]["sigma_model"] = history.history
            else:
                self.histories[key]["sigma_model"] = {
                    loss_key: loss_value + history.history[loss_key]
                    for loss_key, loss_value in self.histories[key][
                        "sigma_model"
                    ].items()
                }
            self.parameters[key]["actual_epochs_sigma"] = len(
                self.histories[key]["sigma_model"]["loss"]
            )

        print()

    #%%
    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""
        super().reset_attributes()
        self.mean_models = OrderedDict()
        self.sigma_models = OrderedDict()

    #%%
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
                history["sigma_model"].get("loss", None),
                label="NOMU DJ " + key[-1] + r": $\hat{r}_f$ loss",
                linestyle="dotted",
            )
            plt.plot(
                history["mean_model"].get("loss", None),
                label="NOMU DJ " + key[-1] + r": $\hat{f}$ loss",
                color=plt1[0].get_color(),
                linestyle="dashed",
            )
            if history["sigma_model"].get("val_loss", None) is not None:
                plt.plot(history["sigma_model"]["val_loss"])
            if history["mean_model"].get("val_loss", None) is not None:
                plt.plot(history["mean_model"]["val_loss"])
        plt.title("Training History", fontsize=20)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(loc="best", prop={"size": 20})
        plt.grid()
        plt.yscale(yscale)

        if save_only:
            plt.savefig(
                fname=absolutepath + "_NOMU_DJ.pdf", format="pdf", transparent=True
            )
            plt.close()

    # %%
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
            filename = "NOMU_DJ_{}".format(model_number)
            model.save(os.path.join(absolutepath, filename) + ".h5")
            # save histories in pickle file
            with open(
                os.path.join(absolutepath, "NOMU_DJ_{}_hist.pkl".format(model_number)),
                "wb",
            ) as f:
                pickle.dump(self.histories[key], f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(
                    absolutepath, "NOMU_DJ_{}_parameters.pkl".format(model_number)
                ),
                "wb",
            ) as f:
                pickle.dump(self.parameters[key], f)
            f.close()
            model_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "NOMU_DJ_all_parameters.txt"), "w") as f:
            for key, p in self.parameters.items():
                f.write(key + ":\n")
                for k, v in p.items():
                    f.write(k + ": " + str(v) + "\n")
                f.write("\n")
        f.close()
        print("\nModels saved in:", absolutepath)

    # %%
    def load_models(
        self, absolutepath: str, model_numbers: Union[int, List[int]], verbose: int
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
            in ["NOMU_DJ_{}.h5".format(model_number) for model_number in model_numbers]
        ]
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "NOMU_DJ_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        hist_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "NOMU_DJ_{}_hist.pkl".format(model_number)
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
            key = "NOMU_DJ_Neural_Network_{}".format(
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
            if verbose > 0:
                print("\nSummary:")
                print(model_NOMU.summary())
            print()
