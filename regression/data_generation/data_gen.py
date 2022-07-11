# -*- coding: utf-8 -*-
"""

This file contains data generators.

"""

# Libs
import numpy as np, matplotlib.pyplot as plt
import matplotlib.colors
from scipy.stats import ortho_group
from scipy import spatial
from numpy.random import dirichlet

# Own Libs
from algorithms.basic_tools import totalRange
from data_generation.function_library import function_library

# %%
def x_data_gen_Gauss(
    din,
    n_train,
    n_val,
    strongDimensions=5,
    strongEigenValue=0.15,
    weakEigenValue=0.001,
    mean=None,
    EigenValuesVector=None,
):
    if mean is None:
        mean = np.zeros(din)
    if EigenValuesVector is None:
        EigenValuesVector = weakEigenValue * np.ones(din)
        EigenValuesVector[0 : min(strongDimensions, din)] = strongEigenValue
    if din == 1:
        A = np.ones(shape=(1, 1))
    else:
        A = ortho_group.rvs(dim=din)
    cov = A @ np.diag(EigenValuesVector) @ A.T
    return np.random.multivariate_normal(
        mean=mean, cov=cov, size=n_train
    ), np.random.multivariate_normal(mean=mean, cov=cov, size=n_val)


# %%
def x_data_gen_Gauss_f(
    strongDimensions=5,
    strongEigenValue=0.15,
    weakEigenValue=0.001,
    mean=None,
    EigenValuesVector=None,
):
    return lambda din, n_train, n_val: x_data_gen_Gauss(
        din=din,
        n_train=n_train,
        n_val=n_val,
        strongDimensions=strongDimensions,
        strongEigenValue=strongEigenValue,
        weakEigenValue=weakEigenValue,
        mean=mean,
        EigenValuesVector=EigenValuesVector,
    )


# %%
def x_data_gen_ConvexHull(
    din,
    n_train,
    n_val,
    x_min=-1,
    x_max=1,
    method=False,
    batch_size_sampling=1000,
    eps=0,
):

    # if xmin or xmax not a list, create one
    if not isinstance(x_min, list):
        x_min = [x_min for _ in range(din)]
    if not isinstance(x_max, list):
        x_max = [x_max for _ in range(din)]
    if len(x_min) != din or len(x_max) != din:
        raise ValueError(
            f"Make sure dimensions are consistent: din={din}, len(x_min)={len(x_min)}, len(x_max)={len(x_max)}"
        )

    x_train = np.random.rand(n_train, din)
    # UNIFORMLY FROM CONVEX HULL
    if method == "uniform":
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
    elif method == "biased":
        outline = spatial.ConvexHull(x_train)
        u_i = dirichlet([1] * x_train.shape[0], size=n_val)
        x_val = u_i @ x_train
    else:
        raise NotImplementedError(f"Method:{method} not supported yet!")

    # FINAL SCALING
    for i, x in enumerate(x_min):
        x_train[:, i] = x_train[:, i] * (x_max[i] - x) + x
        x_val[:, i] = x_val[:, i] * (x_max[i] - x) + x

    return x_train, x_val


# %%
def x_data_gen_ConvexHull_f(
    x_min=-1, x_max=1, method=False, batch_size_sampling=1000, eps=0
):
    return lambda din, n_train, n_val: x_data_gen_ConvexHull(
        din=din,
        n_train=n_train,
        n_val=n_val,
        x_min=x_min,
        x_max=x_max,
        method=method,
        batch_size_sampling=batch_size_sampling,
        eps=eps,
    )


# %%
def x_data_gen_Uniform(
    din,
    n_train,
    n_val,
    x_min=-1,
    x_max=1,
):

    # if xmin or xmax not a list, create one
    if not isinstance(x_min, list):
        x_min = [x_min for _ in range(din)]
    if not isinstance(x_max, list):
        x_max = [x_max for _ in range(din)]
    if len(x_min) != din or len(x_max) != din:
        raise ValueError(
            f"Make sure dimensions are consistent: din={din}, len(x_min)={len(x_min)}, len(x_max)={len(x_max)}"
        )

    x_train = np.random.rand(n_train, din)
    x_val = np.random.rand(n_val, din)

    # FINAL SCALING
    for i, x in enumerate(x_min):
        x_train[:, i] = x_train[:, i] * (x_max[i] - x) + x
        x_val[:, i] = x_val[:, i] * (x_max[i] - x) + x

    return x_train, x_val


# %%
def x_data_gen_Uniform_f(x_min=-1, x_max=1):
    return lambda din, n_train, n_val: x_data_gen_Uniform(
        din=din, n_train=n_train, n_val=n_val, x_min=x_min, x_max=x_max
    )


# %%
def x_data_gen_deterministicGrid(din, n_train, n_val, x_min=-1, x_max=1):

    # if xmin or xmax not a list, create one
    if not isinstance(x_min, list):
        x_min = [x_min for _ in range(din)]
    if not isinstance(x_max, list):
        x_max = [x_max for _ in range(din)]
    if len(x_min) != din or len(x_max) != din:
        raise ValueError(
            f"Make sure dimensions are consistent: din={din}, len(x_min)={len(x_min)}, len(x_max)={len(x_max)}"
        )

    if din == 1:
        x_train = np.linspace(x_min[0], x_max[0], (n_train)).reshape(-1, 1)
    elif din == 2:
        x = np.linspace(x_min[0], x_max[0], int(np.sqrt(n_train)))
        y = np.linspace(x_min[0], x_max[0], int(np.sqrt(n_train)))
        X, Y = np.meshgrid(x, y)
        x_train = np.vstack([Y.reshape(-1), X.reshape(-1)]).T
    else:
        raise NotImplementedError("Higher dimensions than din=2 not supported!")

    x_val = np.random.rand(n_val, din)

    # FINAL SCALING
    for i, x in enumerate(x_min):
        x_val[:, i] = x_val[:, i] * (x_max[i] - x) + x

    return x_train, x_val


def x_data_gen_deterministicGrid_f(x_min=-1, x_max=1):
    return lambda din, n_train, n_val: x_data_gen_deterministicGrid(
        din=din, n_train=n_train, n_val=n_val, x_min=x_min, x_max=x_max
    )


# %% FUNCTION FOR DATA GENERATION
def data_gen(
    din=2,
    dout=1,
    n_train=16,
    n_val=16,
    x_min=-1,
    x_max=1,
    f_true=function_library("Levy"),
    random=True,
    noise_scale=0.05,
    seed=3 + 2,
    noise_on_validation=1,
    plot=True,
    figsize=(10, 10),
    x_data_gen=None,
):

    """generating data that looks like our 'function' argument
    per default: 2D to 1D, f(x,y) = x**2 + y**2
    Example call of the function:
    x_train, y_train, x_val, y_val = data_gen(din=2, x_data_gen=lambda din, n_train, n_val: x_data_gen_Gauss(din=din, n_train=n_train, n_val=n_val,strongDimensions=1))"""

    np.random.seed(seed=seed)

    if x_data_gen is None:
        # RANDOM
        if random:
            print(
                "This functionality will be removed soon -> use x_data_gen = lambda din, n_train, n_val: x_data_gen_Uniform(din,n_train,n_val,x_min=x_min,x_max=x_max) instead!"
            )
            x_data_gen = lambda din, n_train, n_val: x_data_gen_Uniform(
                din,
                n_train,
                n_val,
                x_min=x_min,
                x_max=x_max,
            )
        # DETERMINISTIC GRID
        else:
            print(
                "This functionality will be removed soon -> use x_data_gen = lambda din, n_train, n_val: x_data_gen_deterministicGrid(din,n_train,n_val,x_min=x_min,x_max=x_max) instead!"
            )
            x_data_gen = lambda din, n_train, n_val: x_data_gen_deterministicGrid(
                din, n_train, n_val, x_min=x_min, x_max=x_max
            )

    x_train, x_val = x_data_gen(din, n_train, n_val)

    # TARGETS
    if noise_scale != 0:
        print(
            "Attention! In old version noise scale was scaled down by a factor of 10 until 2021_10_15. Now noise_scale is the standard deviation of the noise."
        )
    if dout == 1:
        y_train = np.random.normal(scale=1, size=n_train) * noise_scale + f_true(
            x_train
        )
        y_val = np.random.normal(
            scale=1, size=n_val
        ) * noise_scale * noise_on_validation + f_true(x_val)
    else:
        y_train = np.random.normal(
            scale=1, size=(n_train, dout)
        ) * noise_scale + f_true(x_train)
        y_val = np.random.normal(
            scale=1, size=(n_val, dout)
        ) * noise_scale * noise_on_validation + f_true(x_val)
    if noise_scale != 0:
        if noise_on_validation == 1:
            print(
                "Now the validation data also has noise. (If you want to remove it change the argument noise_on_validation to 0)"
            )
        elif noise_on_validation != 1:
            print(
                "The noise of the validation data differs by a factor noise_on_validation=",
                noise_on_validation,
                " from the noise_scale of the training data!!! This is typically not desired, unless you exactly know whart you are doing. If you want the validation data to have the same noise-level as the training data, set noise_on_validation to 1.",
                " In an older Version of this code 'noise_on_validation' was called 'yes'. In most situations one should set noise_on_validation=1 !!!!",
            )
    if plot:
        if din == 1:
            resolution = 1280
            x0min, x0max = totalRange(x_train[:, 0], x_val[:, 0])
            x0minPlot = x0min - 0.05 * (x0max - x0min)
            x0maxPlot = x0max + 0.05 * (x0max - x0min)
            x0 = np.linspace(x0minPlot, x0maxPlot, resolution).reshape(-1, 1)
            plt.figure(figsize=figsize)
            plt.plot(x0, f_true(x0), "g-")
            plt.plot(x_train, y_train, "ko", label="Training Data")
            plt.plot(x_val, y_val, "g.", label="Validation Data")
            plt.title(f"Training({n_train}) and Validation({n_val}) Data Points")
            plt.legend()
            plt.show()
        elif din == 2 and dout == 1:
            # resolution=(int(n_train),int(n_train))
            resolution = (64, 64)
            print(f"resolution:{resolution}")

            x0min, x0max = totalRange(x_train[:, 0], x_val[:, 0])
            x1min, x1max = totalRange(x_train[:, 1], x_val[:, 1])
            x0minPlot = x0min - 0.05 * (x0max - x0min)
            x0maxPlot = x0max + 0.05 * (x0max - x0min)
            x1minPlot = x1min - 0.05 * (x1max - x1min)
            x1maxPlot = x1max + 0.05 * (x1max - x1min)
            x0, x1 = np.meshgrid(
                np.linspace(x0minPlot, x0maxPlot, resolution[0]),
                np.linspace(x1minPlot, x1maxPlot, resolution[1]),
            )

            function = np.copy(x0)
            for i in range(resolution[0]):
                for j in range(resolution[1]):
                    function[i, j] = f_true(np.array([[x0[i, j], x1[i, j]]]))
            # y_min=min(np.min(function),np.min(y_train),np.min(y_val))
            # y_max=max(np.max(function),np.max(y_train),np.max(y_val))
            y_min, y_max = totalRange(function, y_train, y_val)
            plt.figure(figsize=figsize)
            tmp_norm = matplotlib.colors.Normalize(vmin=y_min, vmax=y_max)
            plt.contourf(x0, x1, function, levels=20, norm=tmp_norm)
            plt.scatter(
                x_train[:, 0],
                x_train[:, 1],
                marker="o",
                edgecolor="w",
                c=y_train,
                norm=tmp_norm,
                s=80,
                label="Training Data",
            )
            plt.scatter(
                x_val[:, 0],
                x_val[:, 1],
                marker="X",
                edgecolor="k",
                c=y_val,
                norm=tmp_norm,
                s=40,
                label="Validation Data",
            )
            plt.colorbar()
            plt.title(f"Training({n_train}) and Validation({n_val}) Data Points")
            plt.legend()
            plt.show()

    return x_train, y_train, x_val, y_val
