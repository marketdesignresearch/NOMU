# -*- coding: utf-8 -*-
"""

This file implements nll_calibration on a validation set.

"""

# Libs
import numpy as np

# Own Modules
from performance_measures.measures import NLPD

# %%
def nll_calibration(
    model, x_val, y_val, c_grid, add_nlpd_constant=False, model_key=None
):
    if not model_key:
        iterable = model.model_keys
        estimates_val = model.calculate_mean_std(x=x_val)

    else:
        iterable = [model_key]
        estimates_val = model.calculate_mean_std(x=x_val, model_key=model_key)

    NLL_dict = {}

    for key in iterable:
        m, sig = estimates_val[key]

        nll_min = np.inf
        c_min = None

        for c in c_grid:
            nll = NLPD(m, sig, y_val, c, add_constant=add_nlpd_constant)
            if nll < nll_min:
                nll_min = nll
                c_min = c
        NLL_dict[key] = {"c_min": c_min, "nll_min": nll_min}
    return NLL_dict
