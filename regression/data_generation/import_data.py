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
    filename = os.path.join("data_generation","irradiance_data.txt")
    data = pd.read_csv(filename, sep=" ")
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
