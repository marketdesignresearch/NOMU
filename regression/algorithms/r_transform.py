# -*- coding: utf-8 -*-
"""

This file contains a function to create r_transform functions.

"""
import numpy as np


# %% r_transform functions


def r_transform_function(
    r, r_min, r_max, r_transform="custom_min_max", learned_R_TRANSFORM=None
):
    if not (isinstance(r_transform, str)) and not (learned_R_TRANSFORM is None):
        print(
            'learned_R_TRANSFORM is ignored because r_transform="',
            r_transform,
            '".',
        )
    if r_transform == "id":
        return r
    elif r_transform == "custom_min_max":
        return (1 - np.exp(-(r * (r > 0) + r_min) / r_max)) * r_max
    elif r_transform == "relu_cut":
        a = r - r_min
        a[a < 0] = 0
        b = r - r_max
        b[b < 0] = 0
        return r_min + a - b
    else:
        raise NotImplementedError("r_transform {} not implemented.".format(r_transform))
