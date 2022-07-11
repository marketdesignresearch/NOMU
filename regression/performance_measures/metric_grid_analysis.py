# -*- coding: utf-8 -*-
"""
This file contains helper function to analyse metric results.

"""

# Libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# OWN
from algorithms.util import custom_cgrid, pretty_print_dict

# %%


def metric_grid_analysis(results, metric_key="NLPD_grid"):
    seeds = list(results.keys())
    c_grid_params = results[seeds[0]]["c_grid"]
    c_grid = custom_cgrid(
        c_grid_params["grid_min"],
        c_grid_params["grid_max"],
        c_grid_params["steps"],
        c_grid_params["max_power_of_two"],
    )

    # create dict with keys=models and values = metric pandas df (rows=seeds, cols=c_grid)
    dict_metric_mod = {
        k: np.empty(shape=(1, len(c_grid)))
        for k in results[list(results.keys())[0]][metric_key].keys()
    }

    for s in seeds:
        dict_metric_mod = {
            k2: np.append(dict_metric_mod[k2], np.reshape(v2, (1, len(c_grid))), axis=0)
            for k2, v2 in results[s][metric_key].items()
        }

    for k2, v2 in dict_metric_mod.items():
        dict_metric_mod[k2] = pd.DataFrame(data=v2[1:, :], columns=c_grid, index=seeds)

    # create dicts of average metric for each c-value and minimum c-value
    avg_metric_mod = {k: v.mean(axis=0) for k, v in dict_metric_mod.items()}
    min_metric_c = {k: v.iloc[1:].idxmin() for k, v in avg_metric_mod.items()}
    # print('ARGMIN c-values for NLPD-GRID are:')
    print(pretty_print_dict(min_metric_c, printing=False))
    return (dict_metric_mod, min_metric_c)


# %%
def plot_metric_averages(dict_metric, metric_key="NLPD"):
    avg_metric_mod = {k: v.mean(axis=0) for k, v in dict_metric.items()}

    title = f"{metric_key} grid averages:\n"
    plt.figure()
    for k, v in avg_metric_mod.items():
        avg_metric_values = v.iloc[1:]
        c_grid = list(dict_metric[k].T.index)[1:]

        plt.plot(c_grid, avg_metric_values, label=k)
        min_nll = np.min(avg_metric_values)
        min_ind = np.argmin(avg_metric_values)
        min_c = c_grid[min_ind]

        # set title
        title += (
            k
            + ": min"
            + metric_key
            + ": {:.5f}, min "
            + metric_key
            + " factor: {:.2f},\n".format(*[min_nll, min_c])
        )

        plt.text(
            x=min_c * 0.97,
            y=min_nll * 1.1,
            s="{:.2f}".format(min_c),
            fontsize=7,
        )

    plt.legend()
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.grid(which="both")
    plt.ylabel("")
    plt.xlabel("c")
    plt.title(title, fontsize=7)
    plt.tight_layout()
