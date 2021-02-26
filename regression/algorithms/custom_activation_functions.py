# -*- coding: utf-8 -*-

"""

This file contains custom activation functions.

"""

from tensorflow.keras import backend as K
import tensorflow as tf
from typing import Callable

# %%
def softplus_wrapper(min_var: float = 1e-06) -> Callable[[tf.Tensor], float]:

    """Sofplus wrapper with added minimum variance for
    numerical stability, used for class DeepEnsemble and NOMU.

         Arguments
         ----------
         min_var :
             Minimum variance.

         Returns
         -------
         custom_softplus :
             Activation function for tf.layers.

    """

    def custom_softplus(x):
        return K.log(1 + K.exp(x)) + min_var

    return custom_softplus


# %%
def relu_wrapper(min_var: float = 1e-06) -> Callable[[tf.Tensor], float]:

    """Relu wrapper with added minimum variance for
    numerical stability, used for class NOMU.

         Arguments
         ----------
         min_var :
             Minimum variance.

         Returns
         -------
         custom_relu :
             Activation function for tf.layers.

    """

    def custom_relu(x):
        return K.relu(x) + min_var

    return custom_relu
