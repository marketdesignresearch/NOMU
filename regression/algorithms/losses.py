# -*- coding: utf-8 -*-

"""

This file contains loss wrappers and loss functions.

"""

# Libs
import tensorflow as tf
from tensorflow.keras import backend as K
from typing import Callable
import numpy as np


# %%
def r_loss_wrapper(
    flag: bool,
    mu_sqr: float,
    mu_exp: float,
    c_exp: float,
    n_train: int,
    n_aug: int,
    stable_loss: int = 0,
    c_2: float = 1,
    mse: bool = False,
    c_negativ_stable: float = 1,
    c_huber_stable: float = 1,
) -> Callable[[np.array, np.array], float]:

    """Wrapper for the r_loss for our approach. (new)

        Arguments
        ----------
        flag :
            Bool, for data-dependent loss.
        mu_sqr :
            Weight of L2-loss-term of r on training points.
        mu_exp :
            Weight of exp-loss_term.
        c_exp :
            Weight of exponential decay in exp-loss_term.
        n_train :
            Number of training points.
        n_aug :
            Number of augmented points.
    # =============================================================================
    #     stable_aug_loss :
    #         Bool for numerically more stable loss version for x<0. Not used anymore
    # =============================================================================
        stable_loss :
            int for choosing the version of stable_loss
        c_2 :
            Coefficient for squared term in more stable loss.
        mse :
            Bool, if mean (1/n_train) of l2_loss-term should be calculated.
        c_negativ_stable :
            float, constant for stable_loss = 2.
        c_huber_stable :
            float, huber constant for stable_loss = 2.

        Returns
        -------
        custom_loss :
            Data-dependent loss function.

    """
    if n_aug == 0:
        return lambda y_true, y_pred: K.mean(K.square(y_pred))

    def exp_loss_summand(y_true, y_pred):
        return (K.exp(-c_exp * y_pred) * mu_exp) / n_aug

    def stable_exp_loss_summand(y_true, y_pred):
        return (
            c_exp * K.relu(-y_pred)
            + c_2 * K.relu(-y_pred) ** 2
            + K.exp(-c_exp * K.relu(y_pred))
        ) / n_aug

    def stable_exp_loss_summand2(y_true, y_pred):
        return (
            mu_exp
            * (
                c_negativ_stable * c_exp * K.relu(-y_pred)
                + K.exp(-c_exp * K.relu(y_pred))
            )
            / n_aug
        )

    def l2_loss_summand(y_true, y_pred):
        if mse:
            return (K.square(y_pred) * mu_sqr) / n_train
        else:
            return K.square(y_pred) * mu_sqr

    if mse:

        def l2_loss_summand_stable2(y_true, y_pred):
            abs_error = tf.abs(y_pred)
            two_tf = tf.convert_to_tensor(2.0, dtype=abs_error.dtype)
            mu_sqr_divide_n_train_tf = tf.convert_to_tensor(
                mu_sqr / n_train, dtype=abs_error.dtype
            )
            c_huber_stable_tf = tf.convert_to_tensor(
                c_huber_stable, dtype=abs_error.dtype
            )
            return (
                tf.where(
                    abs_error <= c_huber_stable_tf,
                    tf.square(y_pred),
                    two_tf * c_huber_stable_tf * abs_error
                    - tf.square(c_huber_stable_tf),
                )
                * mu_sqr_divide_n_train_tf
            )

    else:

        def l2_loss_summand_stable2(y_true, y_pred):
            y_pred = tf.cast(y_pred, dtype=K.floatx())
            abs_error = tf.abs(y_pred)
            two_tf = tf.convert_to_tensor(2.0, dtype=abs_error.dtype)
            mu_sqr_tf = tf.convert_to_tensor(mu_sqr, dtype=abs_error.dtype)
            c_huber_stable_tf = tf.convert_to_tensor(
                c_huber_stable, dtype=abs_error.dtype
            )
            return (
                tf.where(
                    abs_error <= c_huber_stable_tf,
                    tf.square(y_pred),
                    two_tf * c_huber_stable_tf * abs_error
                    - tf.square(c_huber_stable_tf),
                )
                * mu_sqr_tf
            )

    def custom_stable_loss(y_true, y_pred):
        mask = tf.equal(
            flag, 0
        )  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
        loss = tf.where(
            mask,
            l2_loss_summand(y_true, y_pred),
            stable_exp_loss_summand(y_true, y_pred),
        )
        return K.sum(loss)

    def custom_stable_loss2(y_true, y_pred):
        mask = tf.equal(
            flag, 0
        )  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
        loss = tf.where(
            mask,
            l2_loss_summand_stable2(y_true, y_pred),
            stable_exp_loss_summand2(y_true, y_pred),
        )
        return K.sum(loss)

    def custom_loss(y_true, y_pred):
        mask = tf.equal(
            flag, 0
        )  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
        loss = tf.where(
            mask,
            l2_loss_summand(y_true, y_pred),
            exp_loss_summand(y_true, y_pred),
        )
        return K.sum(loss)

    if stable_loss == 0:
        return custom_loss
    if stable_loss == 1:
        return custom_stable_loss
    if stable_loss == 2:
        return custom_stable_loss2
    else:
        raise NotImplementedError("stable_loss {} not implemented.".format(stable_loss))


# %%
def squared_loss_wrapper(
    flag: bool, n_train: int, n_aug: int, mse: bool = False
) -> Callable[[np.array, np.array], float]:

    """Wrapper for squared loss on training points for our approach. (new)

    Arguments
    ----------
    flag :
        Bool, for data-dependent loss.
    n_train :
        Number of training points.
    n_aug :
        Number of augmented points.
    mse :
        Bool, if mean (1/n_train) of l2_loss-term should be calculated.
    Returns
    -------
    custom_loss :
        Data-dependent loss function.

    """

    def zero_loss(y_true, y_pred):
        return tf.zeros_like(y_true)

    def l2_loss_summand(y_true, y_pred):
        if mse:
            return (K.square(y_true - y_pred)) / n_train
        else:
            return K.square(y_true - y_pred)

    def custom_loss(y_true, y_pred):
        mask = tf.equal(
            flag, 0
        )  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
        loss = tf.where(
            mask, l2_loss_summand(y_true, y_pred), zero_loss(y_true, y_pred)
        )
        return K.sum(loss)

    return custom_loss


# %%
def gaussian_nll(ytrue: tf.Tensor, ypred: tf.Tensor) -> float:

    """Adapted and CHANGED from https://gist.github.com/sergeyprokudin/4a50bf9b75e0559c1fcd2cae860b879e

    Keras implmementation of univariate Gaussian negative loglikelihood loss function.

    Parameters
    ----------
    ytrue: tf.tensor of shape [n_samples, 1]
        ground truth values
    ypreds: tf.tensor of shape [n_samples, 2]
        predicted mu and variance values (e.g. by your neural network)

    Returns
    -------
    neg_log_likelihood : float
        negative loglikelihood summed over samples

    Remarks
    -------
    This loss can then be used as a target loss for any keras model, e.g.:
        model.compile(loss=gaussian_nll, optimizer='Adam').
    Attention keras does mean automatically in loss function.
    """

    mean = ypred[
        :, 0:1
    ]  # TODO: 0:1 is very important -> investigate why ([:,0] does not lead to good results)
    variance = ypred[
        :, 1:
    ]  # TODO: 1: is very important -> investigate why ([:,1] does not lead to good results)

    mse = 0.5 * K.square(ytrue - mean) / variance
    sigma_trace = 0.5 * K.log(variance)
    constant = 0.5 * np.log(2 * np.pi)

    log_likelihood = (
        mse + sigma_trace + constant
    )  # keep the constant to be able to compare the value

    return log_likelihood  # Attention mean is returned (not sum)


# %%
def sum_of_squared_loss(y_true, y_pred):
    return K.square(y_true - y_pred)
