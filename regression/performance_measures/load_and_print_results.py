# -*- coding: utf-8 -*-
"""
This file contains a function to load and print results for (multiple-seed) simulation.
"""
# Libs
import os
import pandas as pd

pd.set_option("display.max_columns", 20)
import pickle
import numpy as np
from scipy.stats import t
from typing import NoReturn

# %%
def load_and_print_results(loadpath: str, seeds: list) -> NoReturn:
    """Loads and prints statistics on measures resulting from regression experiment.

    Arguments
    ----------
    loadpath :
        String giving the path to load pickled results from.
    seeds :
        List of seeds for which the regression experiment had been run.

    """
    # loading results
    with open(os.path.join(loadpath, "results.pkl"), "rb") as f:
        results = pickle.load(f)
    f.close()

    # SUMMARY TO CONSOLE

    ############################## ENTER
    coverage_probability_CI = 0.95
    ##############################

    uQ = (coverage_probability_CI + 1) / 2
    number_of_models = len(results[list(results.keys())[0]]["MW"].keys())
    boxplot_CIs = {
        "AUC": np.zeros((number_of_models, 2)),
        "AUC M-LOG-W": np.zeros((number_of_models, 2)),
        "MIN NLPD": np.zeros((number_of_models, 2)),
    }

    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.width", 200)

    # AUC
    key = list(boxplot_CIs.keys())[0]
    results_auc = {
        k: {k2: v2[0] for k2, v2 in v["MW"].items()} for k, v in results.items()
    }
    pd_auc = pd.DataFrame.from_dict(results_auc)
    pd_auc = pd_auc.T.sort_index()
    print("\n\n{}:".format(key))
    print("Area under the curve (AUC) of ROC-like curve: coverage probability vs. mean width (Measure 1).")
    print("")
    print(
        "#---------------------------------------------------------------------------------"
    )
    print(pd_auc)
    print()
    N = pd_auc.shape[0]
    pd_auc_summary = pd_auc.describe()
    pd_auc_summary.loc["Mean +/-"] = (
        t.ppf(uQ, len(seeds) - 1) * pd_auc_summary.loc["std"] / np.sqrt(N)
    )
    pd_auc_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
        pd_auc_summary.loc["mean"] + pd_auc_summary.loc["Mean +/-"]
    )
    pd_auc_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
        pd_auc_summary.loc["mean"] - pd_auc_summary.loc["Mean +/-"]
    )
    # save LB of CI
    boxplot_CIs[key][:, 0] = np.asarray(
        pd_auc_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))]
    )
    # save UB of CI
    boxplot_CIs[key][:, 1] = np.asarray(
        pd_auc_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))]
    )
    print(pd_auc_summary)
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")

    # AUC-MLW
    key = list(boxplot_CIs.keys())[1]
    results_auc_log = {
        k: {k2: v2[0] for k2, v2 in v["MlogW"].items()} for k, v in results.items()
    }
    pd_auc_log = pd.DataFrame.from_dict(results_auc_log)
    pd_auc_log = pd_auc_log.T.sort_index()
    print("\n\n{} (optional):".format(key))
    print("Area under the curve of ROC-like curve: coverage probability vs. mean LOG(width)")
    print("(Slight adaption of AUC where LOG of width is used instead of width.)")
    print(
        "#---------------------------------------------------------------------------------"
    )
    print(pd_auc_log)
    print()
    N = pd_auc_log.shape[0]
    pd_auc_log_summary = pd_auc_log.describe()
    pd_auc_log_summary.loc["Mean +/-"] = (
        t.ppf(uQ, len(seeds) - 1) * pd_auc_log_summary.loc["std"] / np.sqrt(N)
    )
    pd_auc_log_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
        pd_auc_log_summary.loc["mean"] + pd_auc_log_summary.loc["Mean +/-"]
    )
    pd_auc_log_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
        pd_auc_log_summary.loc["mean"] - pd_auc_log_summary.loc["Mean +/-"]
    )
    # save LB of CI
    boxplot_CIs[key][:, 0] = np.asarray(
        pd_auc_log_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))]
    )
    # save UB of CI
    boxplot_CIs[key][:, 1] = np.asarray(
        pd_auc_log_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))]
    )
    print(pd_auc_log_summary)
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")

    # AUC ARGMAX C
    results_auc_maxFactor = {
        k: {k2: v2[1] for k2, v2 in v["MW"].items()} for k, v in results.items()
    }
    pd_auc_maxFactor = pd.DataFrame.from_dict(results_auc_maxFactor)
    pd_auc_maxFactor = pd_auc_maxFactor.T.sort_index()
    print("\n\nAUC: ARGMAX C")
    print("The calibration parameter c, which achieves 100% coverage probability (c*).")
    print(
        "#---------------------------------------------------------------------------------"
    )
    # print(pd_auc_maxFactor)
    print()
    print(pd_auc_maxFactor.describe())
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")

    # MIN NLPD
    key = list(boxplot_CIs.keys())[2]
    results_nlpd = {
        k: {k2: v2[0] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
    }
    pd_min_nlpd = pd.DataFrame.from_dict(results_nlpd)
    pd_min_nlpd = pd_min_nlpd.T.sort_index()
    print("\n\n{}:".format(key))
    print("Minimum of: average negative (Gaussian) log predictive density (Measure 2).")
    print(
        "#---------------------------------------------------------------------------------"
    )
    print(pd_min_nlpd)
    print()
    N = pd_min_nlpd.shape[0]
    pd_min_nlpd_summary = pd_min_nlpd.describe()
    pd_min_nlpd_summary.loc["Mean +/-"] = (
        t.ppf(uQ, len(seeds) - 1) * pd_min_nlpd_summary.loc["std"] / np.sqrt(N)
    )
    pd_min_nlpd_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
        pd_min_nlpd_summary.loc["mean"] + pd_min_nlpd_summary.loc["Mean +/-"]
    )
    pd_min_nlpd_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
        pd_min_nlpd_summary.loc["mean"] - pd_min_nlpd_summary.loc["Mean +/-"]
    )
    # save LB of CI
    boxplot_CIs[key][:, 0] = np.asarray(
        pd_min_nlpd_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))]
    )
    # save UB of CI
    boxplot_CIs[key][:, 1] = np.asarray(
        pd_min_nlpd_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))]
    )
    print(pd_min_nlpd_summary)
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")

    # NLPD ARGMIN CP
    results_nlpd_argmin_cp = {
        k: {k2: v2[1] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
    }
    pd_nlpd_argmin_cp = pd.DataFrame.from_dict(results_nlpd_argmin_cp)
    pd_nlpd_argmin_cp = pd_nlpd_argmin_cp.T.sort_index()
    print("\n\nNLPD ARGMIN CP:")
    print("Coverage probability at the calibration parameter c that minimizes NLPD(c),")
    print("i.e., coverage probability at the argmin_c NLPD(c).")
    print(
        "#---------------------------------------------------------------------------------"
    )
    # print(pd_nlpd_argmin_cp)
    print()
    print(pd_nlpd_argmin_cp.describe())
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")

    # NLPD ARGMIN C
    results_nlpd_argmin_c = {
        k: {k2: v2[2] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
    }
    pd_nlpd_argmin_c = pd.DataFrame.from_dict(results_nlpd_argmin_c)
    pd_nlpd_argmin_c = pd_nlpd_argmin_c.T.sort_index()
    print("\n\nNLPD ARGMIN C:")
    print("The argmin_c NLPD(c).")
    print(
        "#---------------------------------------------------------------------------------"
    )
    # print(pd_nlpd_argmin_c)
    print()
    print(pd_nlpd_argmin_c.describe())
    print(
        "#---------------------------------------------------------------------------------"
    )

    #
    # save results as txt in folder
    for x, name in zip(
        [
            pd_auc,
            pd_auc_log,
            pd_auc_maxFactor,
            pd_min_nlpd,
            pd_nlpd_argmin_cp,
            pd_nlpd_argmin_c,
        ],
        [
            "pd_auc.txt",
            "pd_auc_log.txt",
            "pd_auc_maxFactor.txt",
            "pd_min_nlpd.txt",
            "pd_nlpd_argmin_cp.txt",
            "pd_nlpd_argmin_c.txt",
        ],
    ):
        x.describe().to_csv(
            os.path.join(loadpath, name), header=True, index=True, sep=" ", mode="w"
        )
    del x
