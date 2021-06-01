# -*- coding: utf-8 -*-
"""

This file contains data generators for augmented data.

"""

# Libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
from numpy.random import dirichlet
from typing import Tuple, Optional, Callable

# Own Modules
from data_generation.function_library import function_library
from data_generation.import_data import data_gen_irradiance

# %% Create augmented data


def generate_augmented_data(
    din: int = 1,
    dout: int = 1,
    n_train: int = 16,
    n_val: int = 16,
    x_min: float = -1,
    x_max: float = 1,
    f_true: Callable[[float], float] = function_library("Levy"),
    random: bool = True,
    noise_scale: float = 0.5,
    seed: int = 3 + 2,
    plot: bool = True,
    noise_on_validation: int = 1,
    figsize: Tuple[float, float] = (16, 9),
    convex_hull_uniform: bool = False,
    convex_hull_biased: bool = False,
    batch_size_sampling: int = 1000,
    eps: float = 0,
    c_aug: float = 10,
    n_aug: int = 2 ** 7,
    random_aug: bool = True,
    x_min_aug: float = -1 - 0.1,
    x_max_aug: float = 1 + 0.1,
    data: Optional[str] = None,
    df: Optional = None,
    train_size: int = 1,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    n_strips: Optional[int] = None,
    lenstrips: Optional[int] = None,
) -> Tuple[np.array, ...]:

    """Returns np.arrays of training and validation data (non-augmented,
    augmented and both), with features augmented by flag
    indicating the type of datapoint (flag==1: artificial data point, flag==0:
    true data point).

            Arguments
            ----------
            din :
                Dimension of input.
            dout :
                Dimension of output.
            n_train :
                Number of training data points.
            n_val :
                Number of validation data points.
            x_min:
                Minimal value for generating training features.
            x_max:
                Maximal value for generating training features.
            f_true:
                Data generating function (callable).
            random:
                If true, training features are sampled randomly in the domain.
            noise_scale:
                Scale parameter for gaussian noise in 'Psychology of Neural
                Network' 's data_gen. ATTENTION: currently (06.10.20) the final
                scale is noise_scale*0.1.
            seed:
                Seed for generating random training features in data_gen.
            plot:
                If true, training and validation data are plotted.
            noise_on_validation:
                If noise_on_validation==1, validation data are generated with same noise scale
                as training data. If noise_on_validation==0, validation data are noiseless.
            figsize:
                Tuple giving range of figure.
            convex_hull_uniform:
                Boolean, should the validation points be sampled uniformly from the convex hull of the training points.
            convex_hull_biased:
                Boolean, should the validation points be sampled from a convex combination from the convex hull of the training points.
            batch_size_sampling:
                Batch size for accept/rejection method in convex_hull sampling
            eps:
                Specifies, tolerance when a validation point is accepted to lie in the convex hull.
            c_aug:
                Value for target of augmented datapoints. Maximal training
                target will be added to it for real data sets.
            n_aug:
                Number of augmented data points.
            random_aug:
                If True, augmented features are sampled randomly in the domain.
            x_min_aug:
                Minimal value for generating augmented features.
            x_max_aug:
                Maximal value for generating augmented features.
            data:
                Name of real data-set.
            df:
                Data frame containing loaded real-world data.
            train_size:
                Proportion of training data (vs. validation data) for real-world
                data generation.
            start:
                Start value for splitting features into training and validation points.
                (real-world interpolation)
            stop:
                Terminal value for splitting features into training and validation points.
                (real-world interpolation)
            n_strips:
                Number of training data strips for splitting features into training and validation points.
                (real-world interpolation)
            lenstrips:
                Length of training data strips for splitting features into training and validation points.
                (real-world interpolation)
            Return
            ----------
            8-Tuple of np.arrays of
            non-augmented and augmented training data concatenated
            non-augmented training data
            augmented training data
            validation data

    """
    # generate training data
    if data == "irradiance":
        x_train, y_train, x_val, y_val, n_train, n_val = data_gen_irradiance(
            df,
            train_size=train_size,
            plot=plot,
            figsize=figsize,
            start=start,
            stop=stop,
            n_strips=n_strips,
            lenstrips=lenstrips,
            seed=seed,
        )
        c_aug += np.max(y_train)
    else:
        x_train, y_train, x_val, y_val = data_gen(
            din=din,
            dout=dout,
            n_train=n_train,
            n_val=n_val,
            x_min=x_min,
            x_max=x_max,
            f_true=f_true,
            random=random,
            noise_scale=noise_scale,
            seed=seed,
            noise_on_validation=noise_on_validation,
            figsize=figsize,
            plot=plot,
            convex_hull_uniform=convex_hull_uniform,
            convex_hull_biased=convex_hull_biased,
            batch_size_sampling=batch_size_sampling,
            eps=eps,
        )
    # augmented random data
    if random_aug:
        x_aug = np.random.uniform(low=x_min_aug, high=x_max_aug, size=(n_aug, din))
    # augmented equidistant data
    else:
        resolution = int(n_aug ** (1 / din))
        x_grid = np.meshgrid(
            *[np.linspace(x_min_aug, x_max_aug, resolution)] * din
        )  # list of length din of arrays of shape (resolution,..., resolution) din times
        x_aug = np.concatenate(
            [np.expand_dims(x, axis=-1) for x in x_grid], axis=-1
        ).reshape((resolution ** din, din))

    y_aug = c_aug * np.ones((n_aug, 1))

    # data prep (concatenate training and random data & add flag)
    x_train = np.concatenate((x_train, np.zeros((n_train, 1))), axis=-1)
    x_aug = np.concatenate((x_aug, np.ones((x_aug.shape[0], 1))), axis=-1)
    x = np.concatenate((x_train, x_aug))
    y = np.concatenate((np.reshape(y_train, (n_train, 1)), y_aug))
    print("input X:\n", pd.DataFrame(x))
    print("\ntarget Y:\n", pd.DataFrame(y))
    if data == "irradiance":
        return (x, y, x_train, y_train, x_aug, y_aug, x_val, y_val, n_train, n_val)
    return (x, y, x_train, y_train, x_aug, y_aug, x_val, y_val)


# %% FUNCTION FOR DATA GENERATION
def data_gen(
    din,
    dout,
    n_train,
    n_val,
    x_min,
    x_max,
    f_true,
    random,
    noise_scale,
    seed,
    noise_on_validation,
    figsize=(10, 10),
    plot=True,
    convex_hull_uniform=False,
    convex_hull_biased=False,
    batch_size_sampling=1000,
    eps=0,
):

    # if xmin or xmax not a list, create one
    if x_min is not list:
        x_min = [x_min for _ in range(din)]
    if x_max is not list:
        x_max = [x_max for _ in range(din)]
    if len(x_min) != din or len(x_max) != din:
        print("Make sure dimensions are consistent")
        return 0, 0, 0, 0
    np.random.seed(seed=seed)
    # INPUTS
    # RANDOM
    if random:
        x_train = np.random.rand(n_train, din)
        # UNIFORMLY FROM CONVEX HULL
        if convex_hull_uniform:
            if convex_hull_biased:
                print(
                    "Ignoring convex_hull_biased arg since convex_hull_uniform was also set to True"
                )
            outline = spatial.ConvexHull(x_train)
            i = 0
            x_val = None
            while i < n_val:
                componentwise_max = np.max(x_train, axis=0)
                componentwise_min = np.min(x_train, axis=0)
                # x_val_tmp = np.random.rand(batch_size_sampling,din)
                x_val_tmp = np.random.uniform(
                    low=componentwise_min,
                    high=componentwise_max,
                    size=(batch_size_sampling, din),
                )
                outside = (
                    outline.equations @ np.c_[x_val_tmp, np.ones(batch_size_sampling)].T
                    > eps
                ).any(0)
                if x_val is None:
                    x_val = x_val_tmp[~outside, :]
                else:
                    x_val = np.concatenate([x_val, x_val_tmp[~outside, :]], axis=0)
                    i = x_val.shape[0]
                    print("Sampled points", i, "/", n_val)
            x_val = x_val[
                :n_val,
            ]  # exactly n_val points
            print("Final validation points", x_val.shape[0])
        # BIASED FROM CONVEX HULL VIA CONVEX COMBINATION
        elif convex_hull_biased:
            outline = spatial.ConvexHull(x_train)
            u_i = dirichlet([1] * x_train.shape[0], size=n_val)
            x_val = u_i @ x_train
            # Here
        # UNIFORMLY
        else:
            x_val = np.random.rand(n_val, din)
        # SCALING
        for i, x in enumerate(x_min):
            x_train[:, i] = x_train[:, i] * (x_max[i] - x) + x
            x_val[:, i] = x_val[:, i] * (x_max[i] - x) + x
    # DETERMINISTIC GRID
    else:
        if din == 1:
            x_train = np.linspace(x_min[0], x_max[0], (n_train)).reshape(-1, 1)
        elif din == 2:
            x = np.linspace(x_min[0], x_max[0], int(np.sqrt(n_train)))
            y = np.linspace(x_min[0], x_max[0], int(np.sqrt(n_train)))
            X, Y = np.meshgrid(x, y)
            x_train = np.vstack([Y.reshape(-1), X.reshape(-1)]).T
        x_val = np.random.rand(n_val, din)
        for i, x in enumerate(x_min):  # scale the columns
            x_val[:, i] = x_val[:, i] * (x_max[i] - x) + x
    # TARGETS
    if dout == 1:
        y_train = 1.0 * (
            np.random.normal(scale=0.1, size=n_train) * noise_scale + f_true(x_train)
        )
        y_val = 1.0 * (
            np.random.normal(scale=0.1, size=n_val) * noise_scale * noise_on_validation
            + f_true(x_val)
        )
    else:
        y_train = 1.0 * (
            np.random.normal(scale=0.1, size=(n_train, dout)) * noise_scale
            + f_true(x_train)
        )
        y_val = 1.0 * (
            np.random.normal(scale=0.1, size=(n_val, dout))
            * noise_scale
            * noise_on_validation
            + f_true(x_val)
        )
    print(
        "Now the validation data also has noise! If you want to remove it change the argument noise_on_validation to 0"
    )
    if plot:
        if din == 1:
            plt.figure(figsize=figsize)
            plt.plot(x_train, y_train, "ko")
            plt.plot(x_val, y_val, "g.")
            plt.show()
        elif din == 2 and dout == 1:
            resolution = (int(n_train), int(n_train))
            xx, yy = np.meshgrid(
                np.linspace(x_min[0] * 1.1, x_max[0] * 1.1, resolution[0]),
                np.linspace(x_min[1] * 1.1, x_max[1] * 1.1, resolution[1]),
            )
            function = np.copy(xx)
            for i in range(resolution[0]):
                for j in range(resolution[1]):
                    function[i, j] = f_true(np.array([[xx[i, j], yy[i, j]]]))
            plt.contourf(xx, yy, function, levels=20)
            plt.colorbar()
            plt.scatter(
                x_val[:, 0], x_val[:, 1], marker="o", edgecolor="w", facecolors="none"
            )  # validation points
            plt.scatter(
                x_train[:, 0],
                x_train[:, 1],
                marker="x",
                edgecolor="k",
                facecolors="k",
                linewidths=100,
                zorder=10,
                s=100,
            )  # training points
            if convex_hull_uniform or convex_hull_biased:
                closed = np.tile(outline.points[outline.vertices], (2, 1))
                closed = closed[: len(closed) // 2 + 1]
                plt.plot(*closed.T, "b")
    return x_train, y_train, x_val, y_val
