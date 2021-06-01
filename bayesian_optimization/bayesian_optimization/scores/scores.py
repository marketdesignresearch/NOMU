# Libs
from typing import Callable
# import tensorflow_probability as tfp
import numpy as np

def gaussian_nll_score(
        ytrue: np.array,
        mu_pred: np.array,
        sigma_pred: np.array,
) -> float:
    # not defined if an entry of sigma_pred is equal to 0, thus set minimum sigma_pred for numerical stability
    sigma_pred += 1e-20

    a = (ytrue - mu_pred) ** 2 / (2 * sigma_pred ** 2) # TODO: check var vs std
    b = np.log(sigma_pred) # TODO: check var vs std
    constant = 0.5 * np.log(2 * np.pi)

    nll = a + b + constant  # keep the constant to be able to compare the value
    return np.average(nll)


# %%
def mse_score(ytrue: np.array,
              mu_pred: np.array,
              ) -> float:
    return np.average((ytrue - mu_pred) ** 2)
