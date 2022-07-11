# -*- coding: utf-8 -*-
"""
This file contains measures for evaluating quality of uncertainty bounds.

"""
# Libs
import numpy as np
from typing import Optional

# %%
def CP(y: np.array, lower: np.array, upper: np.array):
    """Returns coverage probability for given UB and LB of points y.

    Arguments
    ----------
    y :
        np.array of function values from which to compute proportion of covered elements.
    lower :
        np.array of lower bound values (of same shape as upper).
    upper :
        np.array of upper bound values (of same shape as lower).

    Return
    ----------
    Proportion of y-values covered by UB-LB (np.float).

    """
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    return np.mean(np.less_equal(y, upper) & np.greater_equal(y, lower))


# %%
def MW(
    lower: np.array,
    upper: np.array,
    y: Optional[np.array] = None,
    log: Optional[bool] = False,
    log_shift: Optional[float] = 0.001,
):

    """Returns mean (log) width (captured) for given UB and LB.

    Arguments
    ----------
    lower :
        np.array of lower bound values (of same shape as upper).
    upper :
        np.array of upper bound values (of same shape as lower).
    y :
        np.array of function values for mean with captured.
    log :
        flag for calculating mean log width.
    log_shift :
        scalar shift for MlogW: the metric will give mean(log(x+log_shift))
        for an array x of widths.

    Return
    ----------
    Mean (log) width (captured) (np.float).

    """

    # if y values are supplied mean with of captured points is calculated
    if y is not None:

        # reshaping for correct comparison in mask
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            if len(upper.shape) == 1:
                upper = upper.reshape(-1, 1)
                lower = lower.reshape(-1, 1)
        mask = np.squeeze(np.less_equal(y, upper) & np.greater_equal(y, lower))
        # put mw to zero if no points are captured
        w = (upper - lower)[mask]
        if w.size == 0:
            if log:
                return np.log(log_shift)
            else:
                return 0
    else:
        w = upper - lower
    if log:
        w = np.log(w + log_shift)

    return np.mean(w)


# %%
def NLPD(
    m: np.array,
    sig: np.array,
    y_val: np.array,
    c: float,
    add_constant: bool = False,
):
    """Returns average negative log predictive density (captured) for given UB and LB.

    Arguments
    ----------
     m :
        np.array of mean predictions (per validation data point).
    sig :
        np.array of sig predictions (per validation data point).
    y_val :
        np.array of ouptput (target) values of validation data.
    c:
        factor for standard deviation in NLL.
    add_constant:
        bool, should constant of Gaussian NLL be added?
    Return
    ----------
    Average negative log predictive density (captured) (ignoring constants)(np.float).

    """
    # reshape y
    if len(y_val.shape) == 1:
        y_val = y_val.reshape(-1, 1)

    # nll not defined for c=0
    if c == 0:
        return -np.inf

    nll = np.log(c * sig) + ((y_val - m) ** 2) / (2 * (c * sig) ** 2)

    # NEW
    if add_constant:
        nll += 0.5 * np.log(2 * np.pi)

    return np.mean(nll)
