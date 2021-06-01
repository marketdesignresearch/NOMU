# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:22:08 2021

@author: hwutte, jweisstei

This file contains loss wrappers and loss functions.

"""

# Libs
from typing import Callable

# import tensorflow_probability as tfp
import numpy as np

__author__ = "Hanna Wutte, Jakob Weissteiner, Jakob Heiss"
__copyright__ = "Copyright 2020, Pseudo Uncertainty Bounds for Neural Networks"
__license__ = "AGPL-3.0"
__version__ = "0.1.0"
__maintainer__ = "Hanna Wutte, Jakob Weissteiner, Jakob Heiss"
__email__ = "hanna.wutte@math.ethz.ch, weissteiner@ifi.uzh.ch, jakob.heiss@math.ethz.ch"
__status__ = "Dev"
#%%
def gaussian_nll_score(
    ytrue: np.array,
    mu_pred: np.array,
    sigma_pred: np.array,
) -> float:

    # not defined if an entry of sigma_pred is equal to 0, thus set minimum sigma_pred for numerical stability
    sigma_pred += 1e-20

    a = (ytrue - mu_pred) ** 2 / (2 * sigma_pred ** 2)
    b = np.log(sigma_pred)
    constant = 0.5 * np.log(2 * np.pi)

    nll = a + b + constant  # keep the constant to be able to compare the value

    return np.mean(nll)


#%%
def mse_score(
    ytrue: np.array,
    mu_pred: np.array,
) -> float:

    return np.mean((ytrue - mu_pred) ** 2)
