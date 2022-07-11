# -*- coding: utf-8 -*-
"""

This file stores helper functions.

"""

# Libs
import re
from typing import Dict, Tuple
from datetime import timedelta
import numpy as np

# %%
def pretty_print_dict(D: Dict[str, any], printing: bool = True) -> str:

    """Prints a dictionary in a nice way

    Arguments
    ----------
    D :
        Dictionary.
    printing :
        Boolean if it should be printed to console.

    Return
    ----------
    text :
        String representing the dict.

    """

    text = []
    for key, value in D.items():
        if printing:
            print(key, ":  ", value)
        text.append(key + ":  " + str(value) + "\n")
    text = "".join(text)
    return text


# %%
def timediff_d_h_m_s(td: timedelta) -> Tuple[int, int, int, int]:

    """Measures time difference in days, hours,minutes and seconds.

    Arguments
    ----------
    td :
        A timedelta object from the datetime package, representing the difference between two
        datetime.times.

    Return
    ----------
    A tuple representing the td object as days, hours, minutes, and seconds.

    """

    # can also handle negative datediffs
    if (td).days < 0:
        td = -td
        return (
            -(td.days),
            -int(td.seconds / 3600),
            -int(td.seconds / 60) % 60,
            -(td.seconds % 60),
        )
    return td.days, int(td.seconds / 3600), int(td.seconds / 60) % 60, td.seconds % 60


# %%


def update_seed(seed: int, add: int):

    """Updates the a given seed by an increment add.
    Taken from model creator file in Osycholoogy_of_neural_networks.

            Arguments
            ----------
            seed :
                A seed
            add :
                Increment that should be added to the string.

            Return
            ----------
            Incremented seed.

    """

    return [None if seed is None else seed + add][0]


# %%
def key_to_int(key: str) -> int:

    """Finds first integer in a given str key and returns it.

    Arguments
    ----------
    key :
        Key as a string, e.g. UB_Neural_Network_1.

    Returns
    --------
    int_of_key :
        First integer of key.

    """

    int_of_key = int(re.findall(r"\d+", key)[0])
    return int_of_key


# %%
def custom_cgrid(grid_min, grid_max, steps, max_power_of_two):
    """Creates c-grid"""
    factor = (grid_max / grid_min) ** (1 / steps)

    grid = np.array(
        [0]
        + [grid_min * factor ** i for i in range(steps)]
        + [grid_max * 2 ** k for k in range(1, max_power_of_two)]
    )
    return grid
