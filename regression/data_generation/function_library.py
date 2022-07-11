# -*- coding: utf-8 -*-
"""

This file contains a library of synthetic test functions.

"""

import numpy as np
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

#%% helper functions for GaussianBNN
def update_seed(seed, add):
    return [None if seed is None else seed + add][0]


#%% Library of functions


def function_library(function="Step", p=[0.4, 1.0, 2.0, -3.0]):

    """call a function from our very own library, default is 'Levy'"""

    ###########################################################################################################################
    # 1d
    ###########################################################################################################################
    # --------------------------------------------------------------------------------------------------------------------------
    if function == "Step":
        return lambda x: np.squeeze(2.0 * (x[:, 0] < 0) - 1)

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Square":
        return lambda x: 2 * (x[:, 0] ** 2 - 0.5)

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Kink_unscaled":
        return (
            lambda x: p[2] * (x[:, 0] < p[0]) * (x[:, 0] - p[0])
            + p[1]
            + p[3] * (x[:, 0] > p[0]) * (x[:, 0] - p[0])
        )

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Kink":
        extremePoints = function_library(function="Kink_unscaled", p=p)(
            np.array([-1, p[0], 1]).reshape(-1, 1)
        )
        ma = np.max(extremePoints)
        mi = np.min(extremePoints)
        return (
            lambda x: (
                function_library(function="Kink_unscaled", p=p)(x) - (mi + ma) / 2
            )
            * 2
            / (ma - mi)
        )

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Abs":
        return lambda x: np.abs(x[:, 0] - 0.4) * 2 / 1.4 - 1

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Sine1":
        return lambda x: np.sin(10 / (x[:, 0] - 1.2))

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Cubic":
        return lambda x: x[:, 0] ** 3

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Sine2":
        return lambda x: np.sin(-np.pi * 5.5 * x[:, 0]) / 5 + x[:, 0] * 4 / 5

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Forrester":
        return (
            lambda x: (
                (
                    (6 * ((x[:, 0] + 1) * 0.5) - 2) ** 2
                    * np.sin(12 * ((x[:, 0] + 1) * 0.5) - 4)
                )
                + 6.02074
            )
            / (16 * np.sin(8) + 6.02074)
            * 2
            - 1
        )

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Sine3_detrended":
        return lambda x: np.sin(np.exp(2.7 * x[:, 0]) + 0.1) * ((x[:, 0] + 1) / 2) ** (
            0.9
        )

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Sine3":
        return (
            lambda x: (
                0.5 * np.sin(np.exp(2.7 * x[:, 0]) + 0.1) * ((x[:, 0] + 1) / 2) ** (0.9)
                + 0.5 * x[:, 0]
                + 0.25 * (x[:, 0] + 1) ** 2
                - 0.7324646
            )
            / (2.46492193)
            * 2
        )

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Levy":

        def Levy(x):
            w = 1 + (x - 1) / 4
            return (np.sin(np.pi * w)) ** 2 + (w - 1) ** 2 * (
                1 + 1 * (np.sin(2 * np.pi * w)) ** 2
            )

        scale = 10
        ma = Levy(-10)
        mi = 0.0
        return lambda x: (Levy(x[:, 0] * scale) - (ma + mi) * 0.5) * 2 / (ma - mi)

    ###########################################################################################################################
    # (at least) 2d
    ###########################################################################################################################
    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Squared2D":
        return lambda x: (x[:, 0] ** 2 + x[:, 1] ** 2) - 1

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Levy2D":

        def Levy2d(x):
            w = 1 + (x - 1) / 4
            return (
                (np.sin(np.pi * w[:, 0])) ** 2
                + (w[:, 0] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[:, 0] + 1)) ** 2)
                + (w[:, 1] - 1) ** 2 * (1 + 1 * (np.sin(2 * np.pi * w[:, 1])) ** 2)
            )

        scale = 10
        ma = Levy2d(np.array([[-10, -10]]))
        mi = 0.0
        return lambda x: (Levy2d(x * scale) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Gfun2D":

        def Gfun(x):
            return np.multiply(
                (np.abs(4 * x[:, 0] - 2) - 0.5) * 2, (np.abs(4 * x[:, 1] - 2))
            )

        ma = 6
        mi = -2
        return lambda x: (Gfun((x + 1) / 2) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    elif function == "Goldstein2D":

        def Goldstein2d(x):
            return (
                1
                + (x[:, 0] + x[:, 1] + 1) ** 2
                * (
                    19
                    - 14 * x[:, 0]
                    + 3 * x[:, 0] ** 2
                    - 14 * x[:, 1]
                    + 6 * x[:, 0] * x[:, 1]
                    + 3 * x[:, 1] ** 2
                )
            ) * (
                30
                + (2 * x[:, 0] - 3 * x[:, 1]) ** 2
                * (
                    18
                    - 32 * x[:, 0]
                    + 12 * x[:, 0] ** 2
                    + 48 * x[:, 1]
                    - 36 * x[:, 0] * x[:, 1]
                    + 27 * x[:, 1] ** 2
                )
            )

        scale = 2
        ma = 10 ** 6
        mi = 3
        return lambda x: (Goldstein2d(x * scale) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-5,5]^2
    # Global Max: 250 at (5,5)
    # Global Min: -78.332 at (-2.903534,-2.903534)
    elif function == "Styblinski2D":

        def Styblinski2d(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            return 0.5 * (
                (x1 ** 4 - 16 * x1 ** 2 + 5 * x1) + (x2 ** 4 - 16 * x2 ** 2 + 5 * x2)
            )

        scale = 5.0
        ma = 250.0
        mi = -78.332
        return lambda x: (Styblinski2d(scale * x) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-2,2]^2
    # Global Max: 110.5 at (-2,-2)
    # Global Min: 0 at (1,2)
    elif function == "Perm2D":
        # default parameter
        beta_def = 0.5

        def Perm2d(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            return ((1 + beta_def) * (x1 - 1) + (2 + beta_def) * (x2 / 2 - 1)) ** 2 + (
                (1 + beta_def) * (x1 ** 2 - 1)
                + (2 ** 2 + beta_def) * ((x2 / 2) ** 2 - 1)
            ) ** 2

        scale = 2.0
        ma = 110.5
        mi = 0.0
        return lambda x: (Perm2d(scale * x) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-5,10]x[0,15]
    # Global Max: 308.12909601160663 at (-5,0)
    # Global Min: 0.397887 at (-pi,12.275),(pi,2.275),(9.42478,2.475)
    elif function == "Branin2D":
        # default parameter
        a_def = 1
        b_def = 5.1 / (4 * np.pi ** 2)
        c_def = 5 / np.pi
        r_def = 6
        s_def = 10
        t_def = 1 / (8 * np.pi)

        def shiftedBranin2d(x):  # already shifted in the input to [-1,1]**2
            x1 = 15 * ((x[:, 0] + 1) / 2) - 5
            x2 = 15 * ((x[:, 1] + 1) / 2)
            return (
                a_def * (x2 - b_def * x1 ** 2 + c_def * x1 - r_def) ** 2
                + s_def * (1 - t_def) * np.cos(x1)
                + s_def
            )

        ma = 308.12909601160663
        mi = 0.397887
        return lambda x: (shiftedBranin2d(x) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-4.5,4.5]^2
    # Global Max: 181853.61328125 at (-4.5,-4.5)
    # Global Min: 0 at (3,0.5)
    elif function == "Beale2D":

        def Beale2d(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            return (
                (1.5 - x1 + x1 * x2) ** 2
                + (2.25 - x1 + x1 * x2 ** 2) ** 2
                + (2.625 - x1 + x1 * x2 ** 3) ** 2
            )

        scale = 4.5
        ma = 181853.61328125
        mi = 0.0
        return lambda x: (Beale2d(scale * x) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-5,10]^2
    # Global Max: 1102581 at (10,-5)
    # Global Min: 0 at (1,1)
    elif function == "Rosenbrock2D":

        def shiftedRosenbrock2d(x):  # already shifted in the input to [-1,1]**2
            x1 = 15 * ((x[:, 0] + 1) / 2) - 5
            x2 = 15 * ((x[:, 1] + 1) / 2) - 5
            return 100 * (x2 - x1 ** 2) ** 2 + (x1 - 1) ** 2

        ma = 1102581.0
        mi = 0.0
        return lambda x: (shiftedRosenbrock2d(x) - (ma + mi) * 0.5) * 2 / (ma - mi)
    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-2,2]^2
    # Global Max: 9.866666666666665 at (-2,-2) and (2,2)
    # Global Min: 0 at (0,0)
    elif function == "Camel2D":

        def Camel2d(x):
            x1 = x[:, 0]
            x2 = x[:, 1]
            return 2 * x1 ** 2 - 1.05 * x1 ** 4 + x1 ** 6 / 6 + x1 * x2 + x2 ** 2

        scale = 2
        ma = 9.866666666666665
        mi = 0.0
        return lambda x: (Camel2d(scale * x) - (ma + mi) * 0.5) * 2 / (ma - mi)

    # --------------------------------------------------------------------------------------------------------------------------
    # Original input: [-15,-5] x [-3,3]
    # Global Max: 229.178784747792 at (-15,-3)
    # Global Min: 0 at (-10,1)
    elif function == "Bukin2D":

        def shiftedBukin2d(x):  # already shifted in the input to [-1,1]**2
            x1 = 10 * ((x[:, 0] + 1) / 2) - 15
            x2 = 3 * x[:, 1]
            return 100 * np.sqrt(np.abs(x2 - 0.01 * x1 ** 2)) + 0.01 * np.abs(x1 + 10)

        ma = 229.178784747792
        mi = 0.0
        return lambda x: (shiftedBukin2d(x) - (ma + mi) * 0.5) * 2 / (ma - mi)
    # --------------------------------------------------------------------------------------------------------------------------
    # BNNs withstandard Gaussian prior on weights
    elif function == "GaussianBNN":
        layers = p.get("layers", (1, 2 ** 10, 2 ** 11, 2 ** 10, 1))
        print(f"layers:{layers}")
        seed = p.get("seed", 1)
        scaled = p.get("scaled", True)
        print(f"scaled:{scaled}")

        def gaussianbnn(layers, seed):
            # input layer
            x_input = Input(shape=(layers[0],), name="gaussian_bnn_input_layer")

            # first hidden layer
            l = Dense(
                layers[1],
                activation="relu",
                name="gaussian_bnn_hidden_layer_{}".format(1),
                kernel_initializer=RandomNormal(mean=0, stddev=1, seed=seed),
                bias_initializer=RandomNormal(
                    mean=0, stddev=1, seed=update_seed(seed, 1)
                ),
            )(x_input)
            print(f"Seed_Layer_1_Kernel: {seed}")
            print(f"Seed_Layer_1_Bias  : {update_seed(seed,1)}")
            # hidden layers
            for i, n in enumerate(layers[2:-1]):
                l = Dense(
                    n,
                    activation="relu",
                    name="gaussian_bnn_hidden_layer_{}".format(i + 2),
                    kernel_initializer=RandomNormal(
                        mean=0, stddev=1, seed=update_seed(seed, 2 * i + 2)
                    ),
                    bias_initializer=RandomNormal(
                        mean=0, stddev=1, seed=update_seed(seed, 2 * i + 3)
                    ),
                )(l)
                print(f"Seed_Layer_{i+2}_Kernel: {update_seed(seed,2*i+2)}")
                print(f"Seed_Layer_{i+2}_Bias  : {update_seed(seed,2*i+3)}")
            # output layer
            x_output = Dense(
                layers[-1],
                activation="linear",
                name="gaussian_bnn_output_layer",
                kernel_initializer=RandomNormal(
                    mean=0, stddev=1, seed=update_seed(seed, 2 * (i + 1) + 2)
                ),
                bias_initializer=RandomNormal(
                    mean=0, stddev=1, seed=update_seed(seed, 2 * (i + 1) + 3)
                ),
            )(l)
            print(f"Seed_Layer_{i+3}_Kernel: {update_seed(seed,2*(i+1)+2)}")
            print(f"Seed_Layer_{i+3}_Bias  : {update_seed(seed,2*(i+1)+3)}")
            gaussianbnn_tfmodel = Model(inputs=[x_input], outputs=x_output)

            print("\nArchitecture")
            print(gaussianbnn_tfmodel.summary())
            f_gaussian_bnn = lambda x: gaussianbnn_tfmodel.predict(x).reshape(
                -1,
            )
            if scaled:
                # Scaling to \approx Y=[xmin,xmax]
                xmin = p.get("xmin", -1)
                print(f"xmin:{xmin}")
                xmax = p.get("xmax", 1)
                print(f"xmax:{xmax}")
                resolution = p.get("resolution", 1000)
                print(f"resolution:{resolution}")
                X = np.linspace(start=xmin, stop=xmax, num=resolution).reshape(-1, 1)
                pred_X = f_gaussian_bnn(X)
                ma = max(pred_X)
                mi = min(pred_X)
                return lambda x: (f_gaussian_bnn(x) - (ma + mi) * 0.5) * 2 / (ma - mi)
            else:
                return f_gaussian_bnn

        return gaussianbnn(layers=layers, seed=seed)
    # --------------------------------------------------------------------------------------------------------------------------
    else:
        raise NotImplementedError("function not (yet) in library!")
