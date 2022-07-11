# -*- coding: utf-8 -*-
"""

This file contains the model class GaussianProcess

"""
# Libs
from collections import OrderedDict
from itertools import product
from datetime import datetime
import numpy as np
import os
import pickle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
import re
from typing import NoReturn, Union, List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler


# Own Modules
from algorithms.util import pretty_print_dict, timediff_d_h_m_s

# %% Class that stores parameters for BayesByBackropagation Approach: BNNs


class GaussianProcess:

    """
    Gaussian Processes (GP).

    ...

    Attributes
    ----------
    models : OrderedDict
        Tf.keras models.
    parameters : OrderedDict
        Parametersdict for each model.
    initial_kernels : OrderedDict
        Initial kernels for each model.
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
        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.initial_kernels = OrderedDict()
        self.model_keys = []
        self.scaler_input = StandardScaler()
        self.scaler_target = StandardScaler()

    def set_parameters(
        self,
        kernel: Union[str, List[str]],
        whitekernel: Union[bool, List[bool]],
        constant_value: Union[float, List[float]],
        constant_value_bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        length_scale: Union[float, List[float]],
        length_scale_bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        noise_level: Union[float, List[float]],
        noise_level_bounds: Union[Tuple[float, float], List[Tuple[float, float]]],
        alpha: Union[float, List[float]],
        n_restarts_optimizer: Union[int, List[int]],
        optimizer: Union[str, List[str]] = "fmin_l_bfgs_b",
        normalize_y: Union[bool, List[bool]] = True,
        copy_X_train: Union[bool, List[bool]] = True,
        random_state: Optional[Union[int, List[int]]] = None,
        std_min: Optional[Union[float, List[float]]] = 0,
        normalize_data: Optional[Union[bool, List[bool]]] = False,
    ) -> NoReturn:

        """Sets the attributes of the class GaussianProcess.

        Arguments (Note: for each argument also multiple can be specified at once using a list)
        ----------
        kernel : str
            Kernel of GP (currently only rbf)
        whitekernel : bool
             Noise kernel.
        constant_value:  float
            Multiplicative constant factor for scaling the GP kernel.
        constant_value_bounds : Tuple
            Bounds for hyperparameter optimization of constant_value.
        length_scale : float
            Length scale of rbf kernel
        length_scale_bounds : Tuple
            Bounds for hyperparameter optimization of length_scale.
        noise_level : float
            Noise level for PIs (only in combination with whitekernel)
        noise_level_bounds : Tuple
            Bounds for hyperparameter optimization of noise_level.
        alpha : float
            Noise level for CIs
        n_restarts_optimizer : int
            Number of restarts for hyperparameter optimization.
        optimizer : str
            Optimizer used for hyperparameter optimization
        normalize_y : bool
            For normalyzing the input targets.
        copy_X_train : bool
            For copying the input features.
        random_state : int
            Determines random number generation used to initialize the centers.
        std_min : float
            Minimum predictied std for numerical stability in AUC plots.
        normalize_data : bool
                If true, data is normalized s.t. mean=0, std=1

        """

        # set parameters
        parameter_keys = [
            "kernel",
            "whitekernel",
            "constant_value",
            "constant_value_bounds",
            "length_scale",
            "length_scale_bounds",
            "noise_level",
            "noise_level_bounds",
            "alpha",
            "n_restarts_optimizer",
            "optimizer",
            "normalize_y",
            "copy_X_train",
            "random_state",
            "std_min",
            "normalize_data",
        ]
        if not isinstance(kernel, list):
            kernel = [kernel]
        if not isinstance(whitekernel, list):
            whitekernel = [whitekernel]
        if not isinstance(constant_value, list):
            constant_value = [constant_value]
        if not isinstance(constant_value_bounds, list):
            constant_value_bounds = [constant_value_bounds]
        if not isinstance(length_scale, list):
            length_scale = [length_scale]
        if not isinstance(length_scale_bounds, list):
            length_scale_bounds = [length_scale_bounds]
        if not isinstance(noise_level, list):
            noise_level = [noise_level]
        if not isinstance(noise_level_bounds, list):
            noise_level_bounds = [noise_level_bounds]
        if not isinstance(alpha, list):
            alpha = [alpha]
        if not isinstance(n_restarts_optimizer, list):
            n_restarts_optimizer = [n_restarts_optimizer]
        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        if not isinstance(normalize_y, list):
            normalize_y = [normalize_y]
        if not isinstance(copy_X_train, list):
            copy_X_train = [copy_X_train]
        if not isinstance(random_state, list):
            random_state = [random_state]
        if not isinstance(std_min, list):
            std_min = [std_min]
        if not isinstance(normalize_data, list):
            normalize_data = [normalize_data]
        parameters_values = list(
            product(
                kernel,
                whitekernel,
                constant_value,
                constant_value_bounds,
                length_scale,
                length_scale_bounds,
                noise_level,
                noise_level_bounds,
                alpha,
                n_restarts_optimizer,
                optimizer,
                normalize_y,
                copy_X_train,
                random_state,
                std_min,
                normalize_data,
            )
        )

        parameters = [OrderedDict(zip(parameter_keys, x)) for x in parameters_values]
        self.model_keys = [
            "Gaussian_Process_{}".format(i + 1) for i in range(len(parameters))
        ]
        # Set Attributes
        i = 0
        for key in self.model_keys:
            self.parameters[key] = parameters[i]
            self.models[key] = None
            self.initial_kernels[key] = None
            i += 1

    def initialize_models(self, verbose: int = 0) -> NoReturn:

        """Initializes the kernels for the GPs.

        Arguments
        ----------
        verbose :
            Verbosity level.

        """

        print("\nInitialize the following Gaussian Processes:")
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
            if p["kernel"] == "rbf":
                if p["whitekernel"]:
                    K = C(
                        constant_value=p["constant_value"],
                        constant_value_bounds=p["constant_value_bounds"],
                    ) * RBF(
                        length_scale=p["length_scale"],
                        length_scale_bounds=p["length_scale_bounds"],
                    ) + WhiteKernel(
                        noise_level=p["noise_level"],
                        noise_level_bounds=p["noise_level_bounds"],
                    )
                else:
                    K = C(
                        constant_value=p["constant_value"],
                        constant_value_bounds=p["constant_value_bounds"],
                    ) * RBF(
                        length_scale=p["length_scale"],
                        length_scale_bounds=p["length_scale_bounds"],
                    )
            else:
                raise NotImplementedError(
                    "Kernel {} not implemented".format(p["kernel"])
                )
            self.initial_kernels[key] = K
        print()

    def compile_models(self, verbose: int = 0) -> NoReturn:

        """Compiles the GPs.

        Arguments
        ----------
        verbose:
            Verbosity level.

        """

        print("\nCompile the following Gaussian Processes:")
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
        for key, kernel in self.initial_kernels.items():
            print(key)
            p = self.parameters[key]
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=p["alpha"],
                optimizer=p["optimizer"],
                n_restarts_optimizer=p["n_restarts_optimizer"],
                normalize_y=p["normalize_y"],
                copy_X_train=p["copy_X_train"],
                random_state=p["random_state"],
            )
            self.models[key] = gp
        print()

    def fit_models(self, x: np.array, y: np.array, verbose: int = 0) -> NoReturn:

        """Fits the GPs to specified data.

        Arguments
        ----------
        x :
            input data (features)
        y :
            output data (targets).
        verbose :
            Level of verbosity.

        """

        print("\nFit the following Gaussian Processes:")
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
        for key, model in self.models.items():
            p = self.parameters[key]

            if p["normalize_data"]:
                print("Fit function: Fit & Transform x-train...")
                self.scaler_input.fit(x)
                x = self.scaler_input.transform(x)
                print("Fit function: Fit & Transform y-train...")
                y = np.array(y).reshape(-1, 1)
                self.scaler_target.fit(y)
                y = self.scaler_target.transform(y)

            start = datetime.now()
            self.models[key].fit(x, y)
            end = datetime.now()
            diff = end - start
            print(
                "Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(diff)),
                "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
            )
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
            {'Gaussian_Process_1':[array([[mean_1],[mean_2]], dtype=float32),
                                    array([[std_1],[std_2]], dtype=float32)],
             'Gaussian_process_2':...
            }

        """

        # reshape (n,) -> (n,1) only in 1d
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        predictions = OrderedDict()
        for key, model in self.models.items():
            if self.parameters[key]["normalize_data"]:
                print("Prediction function: Transform x-test...")
                x = self.scaler_input.transform(x)

            mu_pred, std_pred = model.predict(x, return_std=True)

            if self.parameters[key]["normalize_data"]:
                print("Prediction function: Inverse-transform y(x-test)...")
                mu_pred = self.scaler_target.inverse_transform(mu_pred)
                std_pred = std_pred * self.scaler_target.scale_

            mu_pred = np.asarray(mu_pred.reshape(-1, 1), dtype=np.float32)
            std_pred = (
                np.asarray(std_pred.reshape(-1, 1), dtype=np.float32)
                + self.parameters[key]["std_min"]
            )
            predictions[key] = [mu_pred, std_pred]
        return predictions

    def calculate_mean_std(self, x: np.array) -> Dict[str, Tuple[np.array, np.array]]:

        """Calculates estimates of the model prediction and uncertainty for each model on a input point x,
        where model prediction is measured in terms of the mean of the GP
        and uncertainty in terms of the std of the GP.

        Arguments
        ----------
        x :
            input data (features)

        Returns
        -------
        predictions:
            A dictionary that stores the predictions for each model, e.g., for x = np.array([[x_1],[x_x]])
            {'Gaussian_Process_1':(array([[mean_1],[mean_2]], dtype=float32),
                                    array([[std_1],[std_2]], dtype=float32)),
             'Gaussian_Process_2':...
            }

        """

        predictions = self.predict(x)
        estimates = OrderedDict()
        for key, v in predictions.items():
            estimates[key] = v[0], v[1]
        return estimates

    def reset_attributes(self) -> NoReturn:

        """Resets attributes of class."""

        self.models = OrderedDict()
        self.parameters = OrderedDict()
        self.initial_kernels = OrderedDict()

    def save_models(self, absolutepath: str) -> NoReturn:

        """Saves models, parameters of the class instance GP.

        Arguments
        ----------
        absolutepath:
            Absolute path for saving.

        """

        # save gp models
        model_number = 1
        for key, model in self.models.items():
            filename = "GP_{}".format(model_number)
            with open(os.path.join(absolutepath, filename + ".pkl"), "wb") as f:
                pickle.dump(model, f)
            f.close()
            # save parameters in pickle file
            with open(
                os.path.join(absolutepath, "GP_{}_parameters.pkl".format(model_number)),
                "wb",
            ) as f:
                pickle.dump(self.parameters[key], f)
            f.close()
            model_number += 1
        # save parameters in txt file
        with open(os.path.join(absolutepath, "GP_all_parameters.txt"), "w") as f:
            for key, p in self.parameters.items():
                f.write(key + ":\n")
                for k, v in p.items():
                    f.write(k + ": " + str(v) + "\n")
                f.write("\n")
        f.close()
        print("Models saved in:", absolutepath)

    def load_models(
        self, absolutepath: str, model_numbers: Union[int, List[int]], verbose: int
    ) -> NoReturn:

        """Loads models and parameters for specified models via model_numbers
        and sets these values in the class instance GP.

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
            if f in ["GP_{}.pkl".format(model_number) for model_number in model_numbers]
        ]
        parameter_files = [
            f
            for f in os.listdir(absolutepath)
            if f
            in [
                "GP_{}_parameters.pkl".format(model_number)
                for model_number in model_numbers
            ]
        ]
        # load
        print("\nLoading the following models:")
        print(
            "**************************************************************************"
        )
        for model_file, parameter_file in zip(model_files, parameter_files):
            key = "Gaussian_Process_{}".format(int(re.findall(r"\d+", model_file)[0]))
            print(key)
            print("Loading file:", parameter_file)
            with open(os.path.join(absolutepath, parameter_file), "rb") as f:
                self.parameters[key] = pickle.load(f)
            f.close()
            print("Loading file:", model_file)
            with open(os.path.join(absolutepath, model_file), "rb") as f:
                model = pickle.load(f)
            f.close()
            self.models[key] = model
            self.initial_kernels[key] = model.kernel_
            if verbose > 0:
                print("\nSummary:")
                for k, v in model.get_params().items():
                    print(k + ":", v)
            print()
