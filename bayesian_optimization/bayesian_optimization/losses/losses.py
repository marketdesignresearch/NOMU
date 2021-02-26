"""
Some code recycled and adopted with permission from https://gitlab.ethz.ch/jakobwe/Pseudo_Uncertainty_Bounds_for_NNs
"""

# Libs
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class SquaredLossWrapper:
    """Class implementing the Squared Loss
    for the NOMU method based on the augmented data
    """

    def __init__(
        self,
        n_train,
        n_aug,
        mse=False
    ):
        self.n_train = n_train
        self.n_aug = n_aug
        self.mse = mse

    def _zero_loss(self, y_true, y_pred):
        return tf.zeros_like(y_true)

    def _l2_loss_summand(self, y_true, y_pred):
        if self.mse:
            return (K.square(y_true - y_pred)) / self.n_train
        else:
            return K.square(y_true - y_pred)

    def get_loss(self, flag):

        def custom_loss(y_true, y_pred):
            mask = tf.equal(flag, 0.)  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
            loss = tf.where(mask,
                            self._l2_loss_summand(y_true, y_pred),
                            self._zero_loss(y_true, y_pred))
            return K.sum(loss)

        return custom_loss


class RLossWrapper:
    """Class implementing the R-Loss
    decides based in the given parameters which exact loss implementation to return and use
    """

    def __init__(
        self,
        mu_sqr: float,
        mu_abs: float,
        mu_exp: float,
        c_exp: float,
        n_train: float,
        n_aug: float,
        stable_aug_loss=False,
        c_2=1,
        mse=False
    ):
        self.mu_sqr = mu_sqr
        self.mu_abs = mu_abs
        self.mu_exp = mu_exp
        self.c_exp = c_exp
        self.n_train = n_train
        self.n_aug = n_aug
        self.stable_aug_loss = stable_aug_loss
        self.c_2 = c_2
        self.mse = mse

    def _l1_exp_loss_summand(self, y_true, y_pred):
        return (K.abs(y_true - y_pred) * self.mu_abs + K.exp(-self.c_exp * y_pred) * self.mu_exp) / self.n_aug

    def _stable_exp_loss_summand(self, y_true, y_pred):
        return(self.c_exp*K.relu(-y_pred) + self.c_2*K.relu(-y_pred)**2 + K.exp(-self.c_exp*K.relu(y_pred)))/self.n_aug

    def _l2_loss_summand(self, y_true, y_pred):
        if self.mse:
            return (K.square(y_pred)*self.mu_sqr)/self.n_train
        else:
            return K.square(y_pred)*self.mu_sqr

    def get_loss(
            self,
            flag: Layer,
    ):
        if self.n_aug == 0:
            return lambda y_true, y_pred: K.mean(K.square(y_pred))

        def custom_stable_loss(y_true, y_pred):
            mask = tf.equal(flag,
                            0.)  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
            loss = tf.where(mask,
                            self._l2_loss_summand(y_true, y_pred),
                            self._stable_exp_loss_summand(y_true, y_pred))
            return K.sum(loss)

        def custom_loss(y_true, y_pred):
            mask = tf.equal(flag,
                            0.)  # if flag==0 then it is a true datapoint, if flag==1 it is a artificial datapoint
            loss = tf.where(mask,
                            self._l2_loss_summand(y_true, y_pred),
                            self._l1_exp_loss_summand(y_true, y_pred))
            return K.sum(loss)

        if self.stable_aug_loss:
            return custom_stable_loss
        else:
            return custom_loss
