# -*- coding: utf-8 -*-
"""

This file contains simple helper functions.

"""
import numpy as np

# %%


def nestedMax(nestedList):
    if not (isinstance(nestedList, list)):
        return np.max(nestedList)
    else:
        return max([nestedMax(a) for a in nestedList])


def nestedMin(nestedList):
    if not (isinstance(nestedList, list)):
        return np.min(nestedList)
    else:
        return min([nestedMin(a) for a in nestedList])


def totalMax(*args):
    return max([nestedMax(a) for a in args])


def totalMin(*args):
    return min([nestedMin(a) for a in args])


def totalRange(*args):
    return totalMin(*args), totalMax(*args)
