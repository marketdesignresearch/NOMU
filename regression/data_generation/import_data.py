# -*- coding: utf-8 -*-
"""

This file contains
    a) import functions and
    b) data_gen functions that create training/validation sets
       applicable to certain datasets.
"""

# Libs
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import Tuple

# %%


def import_irradiance() -> pd.core.frame.DataFrame:
    """Imports irradiance data frame.


    Returns
    --------
    data :
        pandas.core.frame.DataFrame of solar irradiance data.

    """
    filename = os.path.join("data_generation", "irradiance_data.txt")
    data = pd.read_csv(filename, sep=" ")
    return data


# %%


def import_boston():
    filename = "data_generation/UCI_Datasets/Boston_housing.csv"
    column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]
    data = pd.read_csv(filename, header=None, delimiter=r"\s+", names=column_names)
    return data  # MEDV as target


# %%


def import_concrete():
    filename = "data_generation/UCI_Datasets/Concrete_Data_corrected.xls"
    data = pd.read_excel(filename)
    return data


# %%


def import_energy(target="Heating Load"):
    filename = "data_generation/UCI_Datasets/Energy_efficiency_corrected.xls"
    data = pd.read_excel(filename, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    if target == "Heating Load":  # default target
        return data.iloc[:, :-1]
    elif target == "Cooling Load":
        return pd.concat([data.iloc[:, :-2], data.iloc[:, -1]], axis=1)
    else:
        print(
            "Chosen target not available, choose either Heating Load or Cooling Load."
        )


# %%


def import_kin8nm():
    filename = "data_generation/UCI_Datasets/Kin8nm.csv"
    data = pd.read_csv(filename)
    return data


# %%


def import_naval(target="GT Compressor decay state coefficient"):
    filename = "data_generation/UCI_Datasets/Naval/data.txt"
    column_names = [
        "Lever position",
        "Ship speed",
        "Gas Turbine shaft torque",
        "GT rate of revolutions",
        "Gas Generator rate of revolutions",
        "Starboard Propeller Torque",
        "Port Propeller Torque",
        "Hight Pressure Turbine exit temperature",
        "GT Compressor inlet air temperature",
        "GT Compressor outlet air temperature",
        "HP Turbine exit pressure",
        "GT Compressor inlet air pressure",
        "GT Compressor outlet air pressure",
        "GT exhaust gas pressure",
        "Turbine Injecton Control",
        "Fuel flow",
        "GT Compressor decay state coefficient(Y1)",
        "GT Turbine decay state coefficient(Y2)",
    ]
    data = pd.read_csv(filename, header=None, delimiter=r"\s+", names=column_names)
    if target == "GT Compressor decay state coefficient":  # default target
        return data.iloc[:, :-1]
    elif target == "GT Turbine decay state coefficient":
        return pd.concat([data.iloc[:, :-2], data.iloc[:, -1]], axis=1)
    else:
        print(
            "Chosen target not available, choose either GT Compressor decay state coefficient or GT Turbine decay state coefficient."
        )


# %%


def import_ccpp():
    filename = "data_generation/UCI_Datasets/Combined_Cycle_Power_Plant/Folds5x2_pp.xls"
    data = pd.read_excel(filename)
    return data


# %%


def import_protein():
    filename = "data_generation/UCI_Datasets/Protein.csv"
    data = pd.read_csv(filename)
    data = pd.concat(
        [data.iloc[:, 1:], data.iloc[:, 0]], axis=1
    )  # move target to last clmn
    return data


# %%


def import_wine():
    filename = "data_generation/UCI_Datasets/wine_quality.csv"
    data = pd.read_csv(filename)
    return data


def import_yacht():
    filename = "data_generation/UCI_Datasets/yacht_hydrodynamics.data"
    column_names = [
        "Longitudinal position of the center of buoyancy",
        "Prismatic coefficient",
        "Length-displacement ratio",
        "Beam-draught ratio",
        "Length-beam ratio",
        "Froude number",
        "Residuary resistance per unit weight of displacement",
    ]
    data = pd.read_csv(filename, header=None, delimiter=r"\s+", names=column_names)
    return data


# %%


def import_yearmsd():
    filename = "data_generation/UCI_Datasets/YearPredictionMSD.txt"
    column_names = ["Year (Y)"]
    for i in range(1, 91):
        column_names = column_names + ["x" + str(i)]
    data = pd.read_csv(filename, header=None, names=column_names)
    data = pd.concat(
        [data.iloc[:, 1:], data.iloc[:, 0]], axis=1
    )  # move target to last clmn
    return data


# %%


def data_gen_irradiance(
    data: pd.core.frame.DataFrame,
    train_size: float = 1,
    plot: bool = False,
    figsize: Tuple[float, float] = (10, 10),
    start: float = None,
    stop: float = None,
    n_strips: int = None,
    lenstrips: int = None,
    seed: int = 500,
) -> Tuple[np.array, np.array, np.array, np.array, int, int]:

    """Generates (centred & scaled) training and validation sets from solar irradiance data frame.

    Arguments
    ----------
    data :
        Pandas dataframe containing solar irradiance data.
    train_size :
        Float in [0,1], proportion of data to use as training data in case of proportional training/validation set
    plot :
        If true, training and validation data are plotted.
    figsize :
        Tuple giving range of figure.
     tart:
        Start value for splitting features into training and validation points.
    stop:
        Terminal value for splitting features into training and validation points.
    n_strips:
        Number of training data strips for splitting features into training and validation points.
    lenstrips:
        Length of training data strips for splitting features into training and validation points.
    seed:
        Seed for train_test_split.

    Returns
    -------
    X_train:
        np.array of training input points.
    y_train:
        np.array of training output points.
    X_val:
        np.array of validation input points.
    y_val:
        np.array of validation output points.
    n_train:
        int, number of training data points.
    n_val:
        int, number of validation data points.

    """
    X = np.expand_dims(np.array(data["YEAR"]), -1)
    y = np.array(data["11yrCYCLE+BKGRND"])
    X = ((X - np.min(X)) / (np.max(X) - np.min(X))) * 2 - 1
    y = (y - np.min(y)) / (np.max(y) - np.min(y)) * 2 - 1

    if start is not None:
        freespace = stop - start - n_strips * lenstrips
        spacing = freespace / (n_strips - 1)
        init = np.arange(start, stop, lenstrips + int(spacing))
        val_ind = np.concatenate([np.arange(x, (x + 20)) for x in init])
        mask = np.ones(X.shape[0], dtype=bool)
        mask[val_ind] = False
        X_train, X_val, y_train, y_val = X[mask, :], X[~mask, :], y[mask], y[~mask]

        # split into further test data
        if train_size < 1:
            X_train, X_val_additional, y_train, y_val_additional = train_test_split(
                X_train, y_train, shuffle=True, train_size=train_size, random_state=seed
            )
            X_val = np.concatenate([X_val, X_val_additional], axis=0)
            y_val = np.concatenate([y_val, y_val_additional], axis=0)
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
    else:
        if train_size < 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, shuffle=False, train_size=train_size, random_state=seed
            )
            n_train = X_train.shape[0]
            n_val = X_val.shape[0]

        else:
            X_train, X_val, y_train, y_val = X, None, y, None
            n_train = X_train.shape[0]
            n_val = 0
    # plot data
    if plot:
        plt.figure(figsize=figsize)
        plt.grid()
        plt.plot(X_train, y_train, "ko", markersize=1.5, label="Training Data")
        if train_size < 1 or start is not None:
            plt.plot(X_val, y_val, "go", markersize=1.5, label="Validation Data")
        plt.legend()
        plt.show()
    return (X_train, y_train, X_val, y_val, n_train, n_val)


# %% generates training & test splits from UCI data sets
# %% test data is set to be the middle third w.r. to gap_dim (the Gap)
# %% (No normalization or scaling.)
# ---
# Input:
# data = pandas dataframe containing UCI data set
# gap_dim = input dim according to which the gap is chosen
# plot = bool, if split should be displayed by plots
# Output:
# X_train, y_train, X_test, y_test, n_train, n_test
# ---


def data_gen_uci_gap(data, gap_dim, plot=False):
    # Gap-size is 1/3 according to Foong, Li, HL, Turner 2019

    n = data.shape[0]
    d = data.shape[1] - 1
    split_size = int(n / 3)
    train_set_flags = np.array([True] * n)

    indices = np.argsort(
        data.iloc[:, gap_dim]
    )  # sorting indexes according to gap_dim attribute
    gap_indices = indices[split_size:-split_size]
    train_set_flags[gap_indices] = False
    if plot == True:
        plot_train_set_flag = np.array([True] * n)
        plot_train_set_flag[gap_indices] = False
        plt.plot(
            data.iloc[plot_train_set_flag, gap_dim],
            data.iloc[plot_train_set_flag, -1],
            linestyle="",
            marker=".",
            markersize=3,
            label="Training",
        )
        plt.plot(
            data.iloc[~plot_train_set_flag, gap_dim],
            data.iloc[~plot_train_set_flag, -1],
            linestyle="",
            marker=".",
            markersize=3,
            label="Validation/Gap",
        )
        plt.legend()
        plt.title("Gap split according to input dimension " + str(gap_dim))
        plt.show()

    y_train = data.iloc[train_set_flags, -1]
    y_test = data.iloc[~train_set_flags, -1]
    X_train = data.iloc[train_set_flags, :-1]
    X_test = data.iloc[~train_set_flags, :-1]

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if plot == True:
        for d_idx in range(d):
            plt.plot(
                X_train.iloc[:, d_idx],
                y_train,
                linestyle="",
                marker=".",
                markersize=3,
                label="Training",
            )
            plt.plot(
                X_test.iloc[:, d_idx],
                y_test,
                linestyle="",
                marker=".",
                markersize=3,
                label="Validation/Gap",
            )
            plt.legend()
            plt.title(
                "Final split, target vs input dim "
                + str(d_idx)
                + " ("
                + str(100 * n_train / n)[:4]
                + "% training)"
            )
            plt.show()

    print("UCI-Gap: Percentage of training data is " + str(100 * n_train / n)[:4])
    return (X_train, y_train, X_test, y_test, n_train, n_test)


# %% generates training and test sets from UCI data sets (w.o. normalization or scaling)
# %% test data is sampled uniformly over full range
# ---
# Input:
# data = pandas dataframe containing UCI data set
# plot = bool, if split should be displayed by plots
# seed for train/val-test split
# Output:
# X_train, y_train, X_test, y_test, n_train, n_test
# ---


def data_gen_uci(data, plot=False, seed=1):
    # According to Hernandez-Lobato, Adams 2015 Sec. 5.1
    test_size = 0.1

    n = data.shape[0]
    d = data.shape[1] - 1

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if plot == True:
        for gap_dim in range(d):
            plt.plot(
                X_train.iloc[:, gap_dim],
                y_train,
                linestyle="",
                marker=".",
                markersize=3,
                label="Training",
            )
            plt.plot(
                X_test.iloc[:, gap_dim],
                y_test,
                linestyle="",
                marker=".",
                markersize=3,
                label="Validation/Gap",
            )
            plt.legend()
            plt.title(
                "Final split, target vs input dim "
                + str(gap_dim)
                + " ("
                + str(100 * n_train / n)[:4]
                + "% training)"
            )
            plt.show()

    print(
        "Train/Validation-Test split: Percentage of training data (including optional validation data) is "
        + str(100 * n_train / n)[:4]
    )
    return (X_train, y_train, X_test, y_test, n_train, n_test)


# %% generates train & test sets from UCI data set with one dim gap plus uniform sampling
# %% w.o. normalization or scaling.
# ---
# Input:
# data = pandas dataframe containing UCI data set
# test_size = float, requested size of test set additional to Gap as fraction
# plot = bool, if split should be displayed by plots
# seed for train/val-test split
# Output:
# X_train, y_train, X_test, y_test, n_train, n_test
# ---


def data_gen_uci_gap_combo(data, gap_dim, rel_extra_test=1 / 5, plot=False, seed=1):
    # Gap-size is 1/3 according to Foong, Li, HL, Turner 2019

    n = data.shape[0]
    d = data.shape[1] - 1
    split_size = int(n / 3)
    train_set_flags = np.array([True] * n)

    indices = np.argsort(
        data.iloc[:, gap_dim]
    )  # sorting indexes according to gap_dim attribute
    gap_indices = indices[split_size:-split_size]
    train_set_flags[gap_indices] = False
    if plot == True:
        plot_train_set_flag = np.array([True] * n)
        plot_train_set_flag[gap_indices] = False
        plt.plot(
            data.iloc[plot_train_set_flag, gap_dim],
            data.iloc[plot_train_set_flag, -1],
            linestyle="",
            marker=".",
            markersize=3,
            label="Training",
        )
        plt.plot(
            data.iloc[~plot_train_set_flag, gap_dim],
            data.iloc[~plot_train_set_flag, -1],
            linestyle="",
            marker=".",
            markersize=3,
            label="Validation/Gap",
        )
        plt.legend()
        plt.title("Gap split according to input dimension " + str(gap_dim))
        plt.show()

    y_train = data.iloc[train_set_flags, -1]
    y_test = data.iloc[~train_set_flags, -1]
    X_train = data.iloc[train_set_flags, :-1]
    X_test = data.iloc[~train_set_flags, :-1]

    # Additional test points
    X_train, X_test_additional, y_train, y_test_additional = train_test_split(
        X_train, y_train, shuffle=True, test_size=rel_extra_test, random_state=seed
    )
    X_test = pd.concat([X_test, X_test_additional], axis=0)
    y_test = pd.concat([y_test, y_test_additional], axis=0)

    n_train = y_train.shape[0]
    n_test = y_test.shape[0]

    if plot == True:
        for d_idx in range(d):
            plt.plot(
                X_train.iloc[:, d_idx],
                y_train,
                linestyle="",
                marker=".",
                markersize=3,
                label="Training",
            )
            plt.plot(
                X_test.iloc[:, d_idx],
                y_test,
                linestyle="",
                marker=".",
                markersize=3,
                label="Validation/Gap",
            )
            plt.legend()
            plt.title(
                "Final split, target vs input dim "
                + str(d_idx)
                + " ("
                + str(100 * n_train / n)[:4]
                + "% training)"
            )
            plt.show()

    print("UCI-Gap: Percentage of training data is " + str(100 * n_train / n)[:4])
    return (X_train, y_train, X_test, y_test, n_train, n_test)


#%%


def uci_data_selector(name):
    if name == "boston":
        return import_boston()
    elif name == "concrete":
        return import_concrete()
    elif name == "energy":
        return import_energy()
    elif name == "kin8nm":
        return import_kin8nm()
    elif name == "naval":
        return import_naval()
    elif name == "ccpp":
        return import_ccpp()
    elif name == "protein":
        return import_protein()
    elif name == "wine":
        return import_wine()
    elif name == "yacht":
        return import_yacht()
    elif name == "yearmsd":
        return import_yearmsd()
    else:
        print("Error: Data set not found in selection of UCI data sets.")
