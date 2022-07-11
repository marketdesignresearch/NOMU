# -*- coding: utf-8 -*-
"""

This file contains data generators for augmented data.

"""

# Libs
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Callable, Optional, Tuple


# Own Modules
from data_generation.data_gen import (
    data_gen,
    x_data_gen_Uniform_f,
)
from data_generation.function_library import function_library
from data_generation.import_data import (
    data_gen_irradiance,
    data_gen_uci,
    data_gen_uci_gap,
    data_gen_uci_gap_combo,
)

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
    figsize: Tuple[float, float] = (10, 10),
    x_data_gen: Callable[
        [float, float, float], Tuple[np.ndarray, np.ndarray]
    ] = x_data_gen_Uniform_f(x_min=-1, x_max=1),
    c_aug: float = 10,
    n_aug: int = 2 ** 7,
    aug_in_training_range: bool = False,
    aug_range_epsilon: float = 0.05,
    random_aug: bool = False,
    data: Optional[str] = None,
    df: Optional[np.ndarray] = None,
    train_size: int = 1,
    start: Optional[float] = None,
    stop: Optional[float] = None,
    n_strips: Optional[int] = None,
    lenstrips: Optional[int] = None,
    val_size: Optional[bool] = 0,
    val_seed: int = 3 + 2,
    test_seed: int = 3 + 2,
    gap_dim: Optional[int] = -1,
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
            x_data_gen:
                callable, function for generating augmented data points.
            c_aug:
                Value for target of augmented datapoints. Maximal training
                target will be added to it for real data sets.
            n_aug:
                Number of augmented data points.
            aug_in_training_range:
                Boolean; if True, augmented data is sampled in trainings data range +-aug_range_epsilon%
            aug_range_epsilon:
                Percentage by which initial augmented data range is expanded (initial augmented data range is either training data range (if aug_in_training_range==True) or [-1, 1] (else))
            random_aug:
                If True, augmented features are sampled randomly in the domain.
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
            val_size:
                float, proportion of training data to be used as validation data. Only relevant for UCI data sets.
            test_seed:
                Seed for generating random test / validation&training split. Only relevant for UCI data sets.
            val_seed:
                Seed for generating random validation/training split. Only relevant for UCI data sets.
            gap_dim:
                Integer, defining the gap dimension. Only relevant for UCI-Gap experiments on UCI data sets.
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
    elif data == "UCI-Gap":
        x_train, y_train, x_test, y_test, n_train, n_test = data_gen_uci_gap(
            df, gap_dim
        )

    elif data == "UCI":
        x_train, y_train, x_test, y_test, n_train, n_test = data_gen_uci(
            df, seed=test_seed
        )

    elif data == "UCI-Combo":
        (
            x_train,
            y_train,
            x_test,
            y_test,
            n_train,
            n_test,
        ) = data_gen_uci_gap_combo(df, seed=test_seed)

    else:
        x_train, y_train, x_val, y_val = data_gen(
            din=din,
            dout=dout,
            n_train=n_train,
            n_val=n_val,
            f_true=f_true,
            noise_scale=noise_scale,
            seed=seed,
            noise_on_validation=noise_on_validation,
            figsize=figsize,
            plot=plot,
            x_data_gen=x_data_gen,
        )
    if not (data is None) and "UCI" in data:
        # Do train/val split
        try:
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=val_size, random_state=val_seed
            )
            n_val = y_val.shape[0]
        except ValueError:
            print(
                "\nAttention: val_size is not a float in the (0,1) range or a feasible integer. Treated as no validation split.\n"
            )
            x_val = None
            y_val = None
            n_val = 0

        n_train = y_train.shape[0]
        c_aug += np.max(y_train)
        print(
            f"Train-Validation split:\n number of training data is {n_train}\n number of validation data is {n_val}"
        )

    # find range to sample augmented data from
    if aug_in_training_range:
        x_min_aug = x_train.min(axis=0)
        x_max_aug = x_train.max(axis=0)
    else:
        x_min_aug = -1
        x_max_aug = 1
    margin = (x_max_aug - x_min_aug) * aug_range_epsilon
    x_min_aug -= margin
    x_max_aug += margin

    # augmented random data
    if random_aug:
        x_aug = np.random.uniform(low=x_min_aug, high=x_max_aug, size=(n_aug, din))
    # augmented equidistant data
    else:
        resolution = n_aug ** (1 / din)
        if resolution % 1 != 0:
            raise ValueError(
                f"n_aug:{n_aug} is not given as x^{din} for some real number x!"
            )
        else:
            resolution = int(resolution)
        x_grid = np.meshgrid(
            *[np.linspace(x_min_aug, x_max_aug, resolution)] * din
        )  # list of length din of arrays of shape (resolution,..., resolution) din times
        x_aug = np.concatenate(
            [np.expand_dims(x, axis=-1) for x in x_grid], axis=-1
        ).reshape((resolution ** din, din))

    y_aug = c_aug * np.ones((n_aug, 1))

    # data prep (concatenate training and random data & add flag)
    x_train = np.concatenate((x_train, np.zeros((n_train, 1))), axis=-1)
    x_aug = np.concatenate((x_aug, np.ones((n_aug, 1))), axis=-1)
    ###Q: Why add extra dim?

    x = np.concatenate((x_train, x_aug))

    # y = np.concatenate((np.reshape(y_train, (n_train,1)), y_aug))
    y = np.concatenate((np.reshape(np.array(y_train), (n_train, 1)), y_aug))

    # print('input X:\n',pd.DataFrame(x))
    # print('\ntarget Y:\n',pd.DataFrame(y))
    if data == "UCI" or data == "UCI-Gap" or data == "UCI-Combo":
        return (
            x,
            y,
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            x_test,
            y_test,
            n_train,
            n_val,
            n_test,
        )
    if data == "irradiance":
        return (
            x,
            y,
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            n_train,
            n_val,
        )
    return (x, y, x_train, y_train, x_aug, y_aug, x_val, y_val)
