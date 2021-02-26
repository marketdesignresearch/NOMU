# -*- coding: utf-8 -*-
"""

This file implements the Class DataGenerator.

"""

# Libs
from tensorflow.keras.utils import Sequence
import numpy as np
from typing import NoReturn, Any


# %% Classes


# ---
# Class for creating customized batches consisting of a batch of original data and all artifical datapoints
# Initialization:
# batch_size = int, size of batch
# x = np.array, input data including artifical data
# y = np.array, output data including artifical data
# n_train = int, number of true data points
# n_aug = int, number of artifical data points
# shuffle = boolen, shhuffling indexes
# ---
class DataGenerator(Sequence):

    """
    Data Generator for NOMU.

    ...

    Attributes
    ----------
    batch_size:
        Size of generated batches.
    x:
        np.array of training input data (features).
    y:
        np.array of training output data (targets).
    n_train:
        Number of training points.
    n_aug:
        Number of augmented/artificial training points.
    MCaug:
        If true, then a new set of augmented data points is sampled for every batch.
    x_min_aug:
        Minimal value for sampling augmented input data (artificial features).
    x_max_aug:
        Maximal value for sampling augmented input data (artificial features).
    joint_loss:
        If true, y/r-architecture is considered for NOMU.
        If false, data is generated for single output (sigma only) for NOMU DJs.
    shuffle:
        If true, indices are shuffled after every epoch.
    din:
        Dimension of input (features).

    Methods
    -------
    on_epoch_end()
        Draws a new reordering of training data indices after an epoch.
    __len__()
        Computes number of batches per epoch (rounded to next lowest integer value).
    __getitem__()
        Creates a batch of training data for NOMU consisting of batch_size training points
        and n_aug augmented points.

    """

    def __init__(
        self,
        batch_size: int,
        x: np.array,
        y: np.array,
        n_train: int,
        n_aug: int,
        MCaug: bool = False,
        x_min_aug: float = -1 - 0.1,
        x_max_aug: float = 1 + 0.1,
        joint_loss: bool = False,
        shuffle: bool = True,
    ) -> NoReturn:

        """Constructor of the class DataGenerator.

        Sets class attributes.
        """

        self.batch_size = batch_size
        self.x = x
        self.y = y
        self.n_train = n_train
        self.n_aug = n_aug
        self.shuffle = shuffle
        self.on_epoch_end()  # shuffle when instantiating
        self.joint_loss = joint_loss
        self.MCaug = MCaug  # sample new augmented data points with every batch?
        self.x_min_aug = x_min_aug  # set range to sample augmented data from
        self.x_max_aug = x_max_aug
        self.din = x.shape[1] - 1  # input dimension (-1 because we get flagged x input)

    def on_epoch_end(self) -> NoReturn:
        """Specifies action performed at the end of each epoch by an instance
        of the class DataGenerator: after each epoch, training data indices
        are shuffled.
        """
        self.indexes = np.arange(self.n_train)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self) -> int:
        """Computes and returns number of batches per epoch (rounded off to next lowest integer value)."""
        # number of batches per epoch
        return int(np.floor(self.n_train / self.batch_size))

    def __getitem__(self, index: int) -> Any:
        """Creates a batch of training data (input and output) for NOMU
        consisting of batch_size training points and n_aug augmented points.
        If self.MCaug == true, a new set of augmented data points
        is randomly drawn in every batch.

        Return
        ----------
        For y/r-architecture:
        2-Tuple of lists of np.arrays for input and output.
        Input: list of length 2: np.array of input data and np.array of flag input
        Output: list of length 2: np.array of output data and np.array of output data

        """
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # batch indices of training batch and all augmented indices
        indexes_aug = np.concatenate(
            (indexes, np.arange(self.n_train, self.n_train + self.n_aug))
        )  # take in every batch all artifical training points

        if self.MCaug:
            # create new augmented random x-data
            x_aug = np.random.uniform(
                low=self.x_min_aug, high=self.x_max_aug, size=(self.n_aug, self.din)
            )
            x_aug = np.concatenate((x_aug, np.ones((self.n_aug, 1))), axis=-1)
            x = np.concatenate((self.x[indexes, :], x_aug))

        else:
            x = self.x[indexes_aug, :]

        y = self.y[
            indexes_aug
        ]  # select targets. (augmented targets are constant and can hence be taken from initial augmented set)

        # version for joint NOMU training:
        if self.joint_loss:
            return [x[:, :-1], x[:, -1]], [y, y]

        # version for dijoint NOMU training, i.e. NOMU_DJs
        return [x[:, :-1], x[:, -1]], y
