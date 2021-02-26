# -*- coding: utf-8 -*-
"""
This file contains plot functions.
"""

# Libs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from datetime import datetime
import matplotlib.lines as mlines
import itertools
import os
import re
from typing import List, Tuple, Optional, NoReturn, Callable, Dict, Union, Iterable

# Own Modules
from performance_measures.metrics import update_metrics_list, calculate_uncertainty_bounds, split_mu_sigma
from algorithms.model_classes.nomu import NOMU
from algorithms.model_classes.nomu_dj import NOMU_DJ
from algorithms.model_classes.mc_dropout import McDropout
from algorithms.model_classes.gaussian_process import GaussianProcess
from algorithms.model_classes.deep_ensemble import DeepEnsemble

# %%
def plot_predictions(
    x_train: np.array,
    y_train: np.array,
    x_aug: np.array,
    y_aug: np.array,
    x_val: Optional[np.array],
    y_val: Optional[np.array],
    f_true: Optional[Callable[[np.array], float]],
    filepath: Optional[str],
    captured_flag: bool,
    static_parameters: List[str],
    # NOMU
    nomu: Optional[NOMU],
    dynamic_parameters_NOMU: List[str],
    bounds_variant_NOMU: str,
    c_NOMU: float,
    # NOMU_DJ
    nomu_dj: Optional[NOMU_DJ],
    dynamic_parameters_NOMU_DJ: List[str],
    bounds_variant_NOMU_DJ: str,
    c_NOMU_DJ: float,
    # GP
    gp: Optional[GaussianProcess],
    dynamic_parameters_GP: List[str],
    bounds_variant_GP: str,
    c_GP: float,
    # DO
    mc_dropout: Optional[McDropout],
    dynamic_parameters_DO: List[str],
    sample_size_DO: int,
    bounds_variant_DO: str,
    c_DO: float,
    # DE
    deep_ensemble: Optional[DeepEnsemble],
    dynamic_parameters_DE: List[str],
    bounds_variant_DE: str,
    c_DE: float,
    #
    radPlot: float = 1,
    save: bool = False,
    markersize: int = 6,
    transparency: float = 0.2,
    linewidth: int = 2,
    resolution: int = 1920,
    extrapolate: bool = False,
    logy_ROC: bool = False,
    linethreshy_ROC: float = 0.1,
    cp_max_ROC: float = 1,
    c_max_ROC: int = 100,
    custom_c_grid_ROC: Optional[Iterable[float]] = None,
    resolution_ROC: float = 0.01,
    interpolation: str = "linear",
    show_details_title: bool = False,
) -> Union[NoReturn, Dict[str, Tuple[float, float, float]]]:

    """Plots the bounds and Roc-curves for specified models (NOMU,GP,DO,BNN,DE).

    Arguments
    ----------
    x_train :
        Input (features) of training points.
    y_train :
        Output (targets) of training points.
    x_aug :
        Input (features) of augmented points.
    y_aug :
        Output (targets) of augmented points.
    x_val :
        Input (features) of validation points.
    y_val :
        Output (targets) of validation points.
    f_true :
        Ground truth function.
    filepath :
        Absolute Path for saving the plots.
    captured_flag :
        For calculation of coverage probability in ROC. Ifset to True, only connsiders by the bounds captured points, i.e.
        points x where LB(x)<=x<=UB(x).
    static_parameters :
        Static parameters across different models for the plot title and info.
    nomu :
        Instance of NOMU class.
    dynamic_parameters_NOMU :
        Dynamic parameters for NOMU models for plot title and info.
    bounds_variant_NOMU :
        Variant for calculation of uncertainty bounds. For NOMU one has the options 'standard': [y+/-c_NOMU*r] or
        'normal': [y+/-normal_quantile[c_NOMU]*r] on how the credible intervals are computed.
    c_NOMU :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU)/2), where Phi denotes the cdf of N(0,1).
    nomu_dj :
        Instance of NOMU_DJ class.
    dynamic_parameters_NOMU_DJ :
        Dynamic parameters for NOMU_DJ models for plot title and info.
    bounds_variant_NOMU_DJ :
        Variant for calculation of uncertainty bounds. For NOMU_DJ one has the options 'standard': [y+/-c_NOMU_DJ*r] or
        'normal': [y+/-normal_quantile[c_NOMU_DJ]*r] on how the credible intervals are computed.
    c_NOMU_DJ :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU_DJ='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU_DJ='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU_DJ] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU_DJ)/2), where Phi denotes the cdf of N(0,1).
    mc_dropout :
        Instance of McDropout class.
    dynamic_parameters_DO :
        Dynamic parameters for McDropout models for plot title and info.
    sample_size_DO :
        Number of samples drawn to estimate mean and std on a given input point x.
    bounds_variant_DO :
        Variant for calculation of uncertainty bounds. For McDropout one has the options 'standard': [mean+/-c_DO*std] or
        'sample': [lower_sample_quantile[c_DO], upper_sample_quantile[c_DO]] on how the credible intervals are computed,
        i.e, lower_sample_quantile[c_GP]/upper_sample_quantile[c_DO] is given F_emp^-1((1-/+c_DO)/2), where F_emp denotes the empirical cdf of a set of samples.
    c_DO :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DO='standard' this is interpreted as a scalar and
        when bounds_variant_DO='sample' this is intepreted as the coverage probability of the (sample) credible interval.
    gp :
        Instance of GaussianProcess class.
    dynamic_parameters_GP :
        Dynamic parameters for GaussianProcess models for plot title and info.
    bounds_variant_GP :
        Variant for calculation of uncertainty bounds. For GaussianProcess one has the options 'standard': [mean+/-c_GP*std] or
        'normal': [mean+/-normal_quantile[c_GP]*std] on how the credible intervals are computed.
    c_GP :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_GP='standard' this is interpreted as a scalar and
        when bounds_variant_GP='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_GP] is
        given for the lower/upper bound as Phi^-1((1-/+c_GP)/2), where Phi denotes the cdf of N(0,1).
    deep_ensemble :
        Instance of DeepEnsemble class.
    dynamic_parameters_DE :
        Dynamic parameters for DeepEnsemble models for plot title and info.
    bounds_variant_DE :
        Variant for calculation of uncertainty bounds. For DeepEnsemble one has the options 'standard': [mean+/-c_DE*std] or
        'normal': [mean+/-normal_quantile[c_DE]*std] on how the credible intervals are computed.
    c_DE :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DE='standard' this is interpreted as a scalar and
        when bounds_variant_DE='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_DE] is
        given for the lower/upper bound as Phi^-1((1-/+c_DE)/2), where Phi denotes the cdf of N(0,1).
    radPlot :
        Box bounds for the plotting area.
    save :
        Bool for saving the plots in the specified filepath.
    markersize :
        Markersize of markers in plots.
    linewidth :
        Linewidth of lines in plots.
    extrapolate :
        Bool for extrapolating by radPlot from the x_train samples. (assumes x_train to be ordered).
    logy_ROC :
        Bool for a log scale of the oc plot.
    log_shift :
        scalar shift for MlogW: the metric will give mean(log(x+log_shift))
        for an array x of widths.
    linethreshy_ROC :
        Factor that determines at which point on the y-axis the scaling should switch from linear to log for the Roc plot.
    c_max_ROC:
        Maximal c-value considered in the ROC in case variant=='standard'.
    custom_c_grid_ROC :
        Grid for c values to create the ROC plot.
    cp_max_ROC:
        Maximal coverage probability considered in the ROC.
    resolution_ROC:
        Resolution of grid of c-values considered in the ROC.
    interpolation :
        Interpolation type for calculating the np.quantile on samples.
    show_details_title :
        Bool for showing details in the plot title.

    Returns (optional)
    ------------------
    return_dict :
         Dict consisting of AUC's, Max std-factor's/prob's and threshhold factor's/prob's in dict for each model.

    """

    if extrapolate:
        start = x_train[0, 0]
        end = x_train[-1, 0] + radPlot
    else:
        start = -radPlot
        end = radPlot
    legend_ltys = []
    legend_labels = []

    # prepare for calculation of ROC
    if x_val is not None:
        n_val = x_val.shape[0]
        ROC_list = []
        label_list = []
        color_list = []

    # if factors are not provided as lists, change to list
    if not isinstance(c_NOMU, list):
        c_NOMU = [c_NOMU]
    if not isinstance(c_NOMU_DJ, list):
        c_NOMU_DJ = [c_NOMU_DJ]
    if not isinstance(c_DO, list):
        c_DO = [c_DO]
    if not isinstance(c_DE, list):
        c_DE = [c_DE]
    if not isinstance(c_GP, list):
        c_GP = [c_GP]

    # (i) set up plot
    plt.figure(figsize=(16, 9))
    p = plt.plot(
        x_train[:, :-1],
        y_train,
        "ko",
        markersize=markersize,
        label="Training Data",
        zorder=5,
    )
    plt.xlim(start, end)
    title = (
        "Static Parameters: "
        + ", ".join([k + ":{}".format(v) for k, v in static_parameters.items()])
        + "\n"
    )
    title_details = ""
    xPlot = np.linspace(start, end, resolution)

    # create concatenated data
    if x_val is not None:
        concatenated_data = np.concatenate((xPlot, np.squeeze(x_val)))
        sorted_index = np.argsort(concatenated_data)
        xPlot = concatenated_data[sorted_index]  # all x points sorted -> for plot

    # (ii) plot true function or validation data
    if f_true is not None:
        p = plt.plot(xPlot, f_true(xPlot.reshape(-1, 1)), color="black")

    # (iii) plot NN uncertainty bounds
    if nomu is not None:
        title_details += "UB Neural Network Parameters\n " + ", ".join(
            [
                k + ":{}".format(v)
                for k, v in nomu.parameters[list(nomu.parameters.keys())[0]].items()
                if k not in dynamic_parameters_NOMU
            ]
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        # estimate mean and std
        if x_val is not None:
            estimates = nomu.calculate_mean_std(x=concatenated_data)
        else:
            estimates = nomu.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        for key, model in nomu.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )

            # plot mean predictions
            plot_nnub = plt.plot(
                xPlot, mu_predictions, linewidth=linewidth, linestyle="-"
            )

            # plot bounds AND uncertainty
            for c in c_NOMU:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_NOMU,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                p = plt.plot(
                    xPlot,
                    B["uncertainty"],
                    linewidth=linewidth,
                    linestyle=":",
                    color=plot_nnub[0].get_color(),
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "NOMU"
                        + key[-1]
                        + ":"
                        + B["label_uncertainty"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(nomu.parameters[key][k])
                                for k in dynamic_parameters_NOMU
                            ]
                        )
                    )
                else:
                    l = "NOMU: " + B["label_uncertainty"]
                legend_ltys.append(p[0])
                legend_labels.append(l)

                fill_nomu = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], (B["bounds"]["Upper Bound"])[::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_nnub[0].get_color(),
                    ec="None",
                    label=B["label_bounds"],
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "NOMU "
                        + key[-1]
                        + r": $\hat{f}\pm$"
                        + B["label_bounds"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(nomu.parameters[key][k])
                                for k in dynamic_parameters_NOMU
                            ]
                        )
                    )
                else:
                    l = "NOMU: " + r" $\hat{f}\pm$" + B["label_bounds"]
                legend_ltys.append((plot_nnub[0], fill_nomu[0]))
                legend_labels.append(l)

            # calculate ROC data on validation set
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=plot_nnub[0].get_color(),
                    variant=bounds_variant_NOMU,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )
    if nomu_dj is not None:
        title_details += "UB Neural Network Disjoint Parameters\n " + ", ".join(
            [
                k + ":{}".format(v)
                for k, v in nomu_dj.parameters[
                    list(nomu_dj.parameters.keys())[0]
                ].items()
                if k not in dynamic_parameters_NOMU_DJ
            ]
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        # estimate mean and std
        if x_val is not None:
            estimates = nomu_dj.calculate_mean_std(x=concatenated_data)
        else:
            estimates = nomu_dj.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        for key, model in nomu_dj.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )

            # plot mean predictions
            plot_nnub_dj = plt.plot(
                xPlot, mu_predictions, linewidth=linewidth, linestyle="-"
            )

            # plot bounds AND uncertainty
            for c in c_NOMU_DJ:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_NOMU_DJ,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                p = plt.plot(
                    xPlot,
                    B["uncertainty"],
                    linewidth=linewidth,
                    linestyle=":",
                    color=plot_nnub_dj[0].get_color(),
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "NOMU_DJ"
                        + key[-1]
                        + ": "
                        + B["label_uncertainty"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(nomu_dj.parameters[key][k])
                                for k in dynamic_parameters_NOMU_DJ
                            ]
                        )
                    )
                else:
                    l = "NOMU_DJ: " + B["label_uncertainty"]
                legend_ltys.append(p[0])
                legend_labels.append(l)

                fill_nomu_dj = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], (B["bounds"]["Upper Bound"])[::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_nnub_dj[0].get_color(),
                    ec="None",
                    label=B["label_bounds"],
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "NOMU_DJ"
                        + key[-1]
                        + r": $\hat{f}\pm$"
                        + B["label_bounds"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(nomu_dj.parameters[key][k])
                                for k in dynamic_parameters_NOMU_DJ
                            ]
                        )
                    )
                else:
                    l = "NOMU_DJ: " + r" $\hat{f}\pm$" + B["label_bounds"]
                legend_ltys.append((plot_nnub_dj[0], fill_nomu_dj[0]))
                legend_labels.append(l)

            # calculate ROC data on validation set
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=plot_nnub_dj[0].get_color(),
                    variant=bounds_variant_NOMU_DJ,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )
    # (iv) plot Gaussian Process (GP) models
    if gp is not None:
        title_details = title_details + "\nGaussian Process Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in gp.parameters[list(gp.parameters.keys())[0]].items()
                    if (k not in dynamic_parameters_GP and v is not None)
                ]
            )
            + "\n"
        )

        # estimate mean and std
        if x_val is not None:
            estimates = gp.calculate_mean_std(x=concatenated_data)
        else:
            estimates = gp.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        for key, model in gp.models.items():
            title_details = (
                title_details
                + key
                + ":  Initial: {} | Optimum: {} | Log-Marginal-Likelihood: {}".format(
                    gp.initial_kernels[key],
                    model.kernel_,
                    round(model.log_marginal_likelihood(model.kernel_.theta), 4),
                )
            )

            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot mean predictions
            plot_gpr = plt.plot(
                xPlot, mu_predictions, linewidth=linewidth, linestyle="-"
            )
            # plot bounds
            for c in c_GP:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_GP,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                fill_gpr = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_gpr[0].get_color(),
                    ec="None",
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "GP "
                        + key[-1]
                        + r": $\hat{f}\pm$"
                        + B["label_bounds"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(gp.parameters[key][k])
                                for k in dynamic_parameters_GP
                            ]
                        )
                    )
                else:
                    l = "GP: " + r" $\hat{f}\pm$" + B["label_bounds"]
                legend_ltys.append((plot_gpr[0], fill_gpr[0]))
                legend_labels.append(l)

            # calculating ROC data on validation set
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=plot_gpr[0].get_color(),
                    variant=bounds_variant_GP,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )
    # (v) plot MC Dropout Models
    if mc_dropout is not None:

        # predict from dropout model
        if x_val is not None:
            predictions = mc_dropout.predict(x=concatenated_data)
        else:
            predictions = mc_dropout.predict(x=xPlot)

        # estimate mean and std
        estimates = mc_dropout.calculate_mean_std(x=None, predictions=predictions)

        # plot each model in class instance
        for key, model in mc_dropout.models.items():
            predictions_plot = predictions[key]

            if x_val is not None:
                # split predictions
                predictions_val = [
                    x[-n_val:, :] for x in predictions_plot
                ]  # dropout predictions on validation set -> for ROC
                predictions_plot = [
                    x[sorted_index] for x in predictions_plot
                ]  # all dropout predictions sorted -> for plot

            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot mean prediction
            plot_dp = plt.plot(xPlot, mu_predictions, linewidth=linewidth)

            # plot bounds
            for c in c_DO:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_DO,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    raw_predictions=predictions_plot,
                    model_key=None,
                )

                fill_dp = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_dp[0].get_color(),
                    ec="None",
                )

                # append legend style and label
                if int(key[-1]) > 1:
                    l = (
                        "MCDO "
                        + key[-1]
                        + r": $\hat{f}\pm$"
                        + B["label_bounds"]
                        + ", ".join(
                            [""]
                            + [
                                k + ": " + str(mc_dropout.parameters[key][k])
                                for k in dynamic_parameters_DO
                            ]
                        )
                    )
                else:
                    l = "MCDO: " + r" $\hat{f}\pm$" + B["label_bounds"]
                legend_ltys.append((plot_dp[0], fill_dp[0]))
                legend_labels.append(l)

        title_details = title_details + "\nMc Dropout Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in mc_dropout.parameters[
                        list(mc_dropout.parameters.keys())[0]
                    ].items()
                    if k not in dynamic_parameters_DO
                ]
            )
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        # calculate ROC data based on alidation data
        if x_val is not None:
            ROC_list, label_list, color_list = update_metrics_list(
                ROC_list=ROC_list,
                label_list=label_list,
                color_list=color_list,
                label=key,
                color=plot_dp[0].get_color(),
                variant=bounds_variant_DO,
                m=mu_predictions_val,
                sig=std_predictions_val,
                y_val=y_val,
                predictions=predictions_val,
                captured_flag=captured_flag,
                c_max=c_max_ROC,
                custom_c_grid=custom_c_grid_ROC,
                cp_max=cp_max_ROC,
                resolution=resolution_ROC,
                interpolation=interpolation,
            )

    # (vii) plot Deep Ensemble Models
    if deep_ensemble is not None:

        # estimate mean and std
        if x_val is not None:
            estimates = deep_ensemble.calculate_mean_std(x=concatenated_data)
        else:
            estimates = deep_ensemble.estimate_models(x=xPlot)

        # plot
        for ensemble_key, ensemble in deep_ensemble.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[ensemble_key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot the mean output
            plot_de = plt.plot(xPlot, mu_predictions, linewidth=linewidth)
            # plot the bounds
            for c in c_DE:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_DE,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                fill_de = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_de[0].get_color(),
                    ec="None",
                )

                # append legend style and label
                if int(ensemble_key[-1]) > 1:
                    l = (
                        "DE "
                        + ensemble_key[-1]
                        + r": $\hat{f}\pm$"
                        + B["label_bounds"]
                        + ", ".join(
                            [""]
                            + [
                                k
                                + ": "
                                + str(deep_ensemble.parameters[ensemble_key][k])
                                for k in dynamic_parameters_DE
                            ]
                        )
                    )
                else:
                    l = "DE: " + r" $\hat{f}\pm$" + B["label_bounds"]
                legend_ltys.append((plot_de[0], fill_de[0]))
                legend_labels.append(l)

            # calculate ROC data based on validation data
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=ensemble_key,
                    color=plot_de[0].get_color(),
                    variant=bounds_variant_DE,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )

        title_details = title_details + "\nDeep Ensemble Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in deep_ensemble.parameters[
                        list(deep_ensemble.parameters.keys())[0]
                    ].items()
                    if k not in dynamic_parameters_DE
                ]
            )
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

    # (viii) finish plot
    # To specify the number of ticks on both or any single axes
    plt.locator_params(axis="y", nbins=5)
    plt.locator_params(axis="x", nbins=5)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(legend_ltys, legend_labels, loc="best", shadow=True, fontsize=20)

    plt.grid(True)
    if show_details_title:
        plt.title(title + title_details, fontsize="small")
    plt.tight_layout()

    # (ix) save plot and title details and ROC
    savepath = None
    if save:

        if filepath is not None:

            # save plot
            if static_parameters.get("random_locations"):
                fname = "_seed{}_".format(
                    static_parameters.get("seed")
                ) + datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            else:
                fname = "_" + datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            savepath = os.path.join(filepath, "Plot_Bounds" + fname + ".pdf")
            plt.savefig(savepath, format="pdf", transparent=True)

            # save info
            savepath = os.path.join(filepath, "Plot_Info" + fname + ".txt")
            with open(savepath, "w") as f:
                f.write(title + title_details)
            f.close()

            # savepath for ROC based on MW
            ROC_plot_path = os.path.join(filepath, "Plot_Roc" + fname)
            plt.close()
        else:
            print("No filepath for saving the bounds plot specified.")
            print("No filepath for saving the info file specified.")
            print(
                "No filepath for saving the ROC plot specified."
            ) if x_val is not None else None

    if x_val is not None:
        return_dict = ROC_plot(
            ROC_list,
            label_list,
            color_list,
            logy=logy_ROC,
            linethreshy=linethreshy_ROC,
            savepath=ROC_plot_path if save else None,
            captured_flag=captured_flag,
            bounds_variant_DO=bounds_variant_DO,
            bounds_variant_GP=bounds_variant_GP,
            bounds_variant_DE=bounds_variant_DE,
            bounds_variant_NOMU=bounds_variant_NOMU,
        )

        return return_dict


# %%
def plot_predictions_2d(
    x_train: np.array,
    y_train: np.array,
    x_aug: np.array,
    y_aug: np.array,
    x_val: Optional[np.array],
    y_val: Optional[np.array],
    f_true: Optional[Callable[[np.array], float]],
    filepath: Optional[str],
    captured_flag: bool,
    static_parameters: List[str],
    # NOMU
    nomu: Optional[NOMU],
    dynamic_parameters_NOMU: List[str],
    bounds_variant_NOMU: str,
    c_NOMU: float,
    # NOMU_DJ
    nomu_dj: Optional[NOMU_DJ],
    dynamic_parameters_NOMU_DJ: List[str],
    bounds_variant_NOMU_DJ: str,
    c_NOMU_DJ: float,
    # GP
    gp: Optional[GaussianProcess],
    dynamic_parameters_GP: List[str],
    bounds_variant_GP: str,
    c_GP: float,
    # DO
    mc_dropout: Optional[McDropout],
    dynamic_parameters_DO: List[str],
    sample_size_DO: int,
    bounds_variant_DO: str,
    c_DO: float,
    # DE
    deep_ensemble: Optional[DeepEnsemble],
    dynamic_parameters_DE: List[str],
    bounds_variant_DE: str,
    c_DE: float,
    #
    radPlot: float = 1,
    save: bool = False,
    markersize: int = 6,
    transparency: float = 0.2,
    linewidth: int = 2,
    resolution: int = 1920,
    extrapolate: bool = False,
    logy_ROC: bool = False,
    linethreshy_ROC: float = 0.1,
    cp_max_ROC: float = 1,
    c_max_ROC: int = 100,
    custom_c_grid_ROC: Optional[Iterable[float]] = None,
    resolution_ROC: float = 0.01,
    interpolation: str = "linear",
    show_details_title: bool = False,
    colorlimits: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (16, 9),
    only_uncertainty: bool = False,
) -> Union[NoReturn, Dict[str, Tuple[float, float, float]]]:

    """Plots the bounds and Roc-curves for specified models (NOMU,GP,DO,BNN,DE).

    Arguments
    ----------
    x_train :
        Input (features) of training points.
    y_train :
        Output (targets) of training points.
    x_aug :
        Input (features) of augmented points.
    y_aug :
        Output (targets) of augmented points.
    x_val :
        Input (features) of validation points.
    y_val :
        Output (targets) of validation points.
    f_true :
        Ground truth function.
    filepath :
        Absolute Path for saving the plots.
    captured_flag :
        For calculation of coverage probability in ROC. Ifset to True, only connsiders by the bounds captured points, i.e.
        points x where LB(x)<=x<=UB(x).
    static_parameters :
        Static parameters across different models for the plot title and info.
    nomu :
        Instance of NOMU class.
    dynamic_parameters_NOMU :
        Dynamic parameters for NOMU models for plot title and info.
    bounds_variant_NOMU :
        Variant for calculation of uncertainty bounds. For NOMU one has the options 'standard': [y+/-c_NOMU*r] or
        'normal': [y+/-normal_quantile[c_NOMU]*r] on how the credible intervals are computed.
    c_NOMU :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU)/2), where Phi denotes the cdf of N(0,1).
    nomu_dj :
        Instance of NOMU_DJ class.
    dynamic_parameters_NOMU_DJ :
        Dynamic parameters for NOMU_DJ models for plot title and info.
    bounds_variant_NOMU_DJ :
        Variant for calculation of uncertainty bounds. For NOMU_DJ one has the options 'standard': [y+/-c_NOMU_DJ*r] or
        'normal': [y+/-normal_quantile[c_NOMU_DJ]*r] on how the credible intervals are computed.
    c_NOMU_DJ :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU_DJ='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU_DJ='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU_DJ] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU_DJ)/2), where Phi denotes the cdf of N(0,1).
    mc_dropout :
        Instance of McDropout class.
    dynamic_parameters_DO :
        Dynamic parameters for McDropout models for plot title and info.
    sample_size_DO :
        Number of samples drawn to estimate mean and std on a given input point x.
    bounds_variant_DO :
        Variant for calculation of uncertainty bounds. For McDropout one has the options 'standard': [mean+/-c_DO*std] or
        'sample': [lower_sample_quantile[c_DO], upper_sample_quantile[c_DO]] on how the credible intervals are computed,
        i.e, lower_sample_quantile[c_GP]/upper_sample_quantile[c_DO] is given F_emp^-1((1-/+c_DO)/2), where F_emp denotes the empirical cdf of a set of samples.
    c_DO :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DO='standard' this is interpreted as a scalar and
        when bounds_variant_DO='sample' this is intepreted as the coverage probability of the (sample) credible interval.
    gp :
        Instance of GaussianProcess class.
    dynamic_parameters_GP :
        Dynamic parameters for GaussianProcess models for plot title and info.
    bounds_variant_GP :
        Variant for calculation of uncertainty bounds. For GaussianProcess one has the options 'standard': [mean+/-c_GP*std] or
        'normal': [mean+/-normal_quantile[c_GP]*std] on how the credible intervals are computed.
    c_GP :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_GP='standard' this is interpreted as a scalar and
        when bounds_variant_GP='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_GP] is
        given for the lower/upper bound as Phi^-1((1-/+c_GP)/2), where Phi denotes the cdf of N(0,1).
    deep_ensemble :
        Instance of DeepEnsemble class.
    dynamic_parameters_DE :
        Dynamic parameters for DeepEnsemble models for plot title and info.
    bounds_variant_DE :
        Variant for calculation of uncertainty bounds. For DeepEnsemble one has the options 'standard': [mean+/-c_DE*std] or
        'normal': [mean+/-normal_quantile[c_DE]*std] on how the credible intervals are computed.
    c_DE :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DE='standard' this is interpreted as a scalar and
        when bounds_variant_DE='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_DE] is
        given for the lower/upper bound as Phi^-1((1-/+c_DE)/2), where Phi denotes the cdf of N(0,1).
    radPlot :
        Box bounds for the plotting area.
    save :
        Bool for saving the plots in the specified filepath.
    markersize :
        Markersize of markers in plots.
    linewidth :
        Linewidth of lines in plots.
    extrapolate :
        Bool for extrapolating by radPlot from the x_train samples. (assumes x_train to be ordered).
    logy_ROC :
        Bool for a log scale of the oc plot.
    linethreshy_ROC :
        Factor that determines at which point on the y-axis the scaling should switch from linear to log for the Roc plot.
    c_max_ROC:
        Maximal c-value considered in the ROC in case variant=='standard'.
    custom_c_grid_ROC :
        Grid for c values to create the ROC plot.
    cp_max_ROC:
        Maximal coverage probability considered in the ROC.
    resolution_ROC:
        Resolution of grid of c-values considered in the ROC.
    interpolation :
        Interpolation type for calculating the np.quantile on samples.
    show_details_title :
        Bool for showing details in the plot title.
    colorlimits :
        Limits for colors for 'contour' - plot.
    figsize :
        Size of figure, e.g. (16,9).
    only_uncertainty :
        Bool for plotting the uncertainty only and not the upper bound of the credible interval.

    Returns (optional)
    ------------------
    return_dict :
         Dict consisting of AUC's, Max std-factor's/prob's and threshhold factor's/prob's in dict for each model.

    """

    # prepare for calculation of ROC
    if x_val is not None:
        n_val = x_val.shape[0]
        ROC_list = []
        label_list = []
        color_list = []

    # (i) set up plot
    din = 2  # 2D
    colorcylcer = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    xPlot = np.meshgrid(
        *[np.linspace(-radPlot, radPlot, resolution)] * din
    )  # list of length din of arrays of shape (resolution,..., resolution) din=2 times
    x_grid = np.concatenate(
        [np.expand_dims(x, axis=-1) for x in xPlot], axis=-1
    ).reshape((resolution ** din, din))
    fig_dict = {}
    title_details_dict = {}

    # create concatenated data
    if x_val is not None:
        concatenated_data = np.concatenate((x_grid, np.squeeze(x_val)), axis=0)

    if colorlimits is None:
        tmpcolornorm = colors.Normalize(
            vmin=np.min(y_train), vmax=max(np.max(y_aug), np.max(y_train))
        )
    else:
        tmpcolornorm = colors.Normalize(vmin=colorlimits[0], vmax=colorlimits[1])

    # (ii) plot True Function
    if f_true is not None:
        title = "Ground Truth Function"
        title_details = "\nStatic Parameters:" + ", ".join(
            [k + ":{}".format(v) for k, v in static_parameters.items()]
        )
        title_details_dict["True_Function"] = title + title_details
        if show_details_title:
            title += title_details
        y = np.copy(xPlot[0])
        for i in range(resolution):
            for j in range(resolution):
                y[i, j] = f_true(np.array([[xPlot[0][i, j], xPlot[1][i, j]]]))
        fig = fig_2d(
            xPlot[0],
            xPlot[1],
            y,
            x_train,
            x_aug,
            y_train,
            linewidth=linewidth,
            markersize=markersize,
            colornorm=tmpcolornorm,
            title=title,
            radPlot=radPlot,
            figsize=figsize,
        )
        fig_dict["True_Function"] = fig

    # (iii) plot NN Uncertainty Bounds
    if nomu is not None:
        estimates = nomu.calculate_mean_std(x=x_grid)  # estimate mean and std

        if x_val is not None:
            estimates_val = nomu.calculate_mean_std(x=x_val)

        # plot each model in class instance
        for key, model in nomu.models.items():
            short_key = "NOMU_{}".format(int(re.findall(r"\d+", key)[0]))
            mu_predictions, std_predictions = estimates[key]
            B = calculate_uncertainty_bounds(
                c=c_NOMU,
                variant=bounds_variant_NOMU,
                mu_predictions=mu_predictions,
                std_predictions=std_predictions,
                model_key=key,
            )
            if only_uncertainty:
                y = B["uncertainty"]
                title = "Plot Shows Uncertainty of " + B["label_uncertainty"]
            else:
                y = B["bounds"]["Upper Bound"]
                title = "Plot Shows Upper Bound of " + B["label_bounds"]

            title_details = "\nStatic Parameters:" + ", ".join(
                [k + ":{}".format(v) for k, v in static_parameters.items()]
            )
            title_details += "\nParameters: " + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in nomu.parameters[list(nomu.parameters.keys())[0]].items()
                ]
            )
            title_details = title_details.replace(", optimizer", "\noptimizer")
            title_details_dict[short_key] = key + title_details

            if show_details_title:
                title += title_details
            fig = fig_2d(
                xPlot[0],
                xPlot[1],
                y.T[0].reshape(resolution, resolution),
                x_train,
                y_train,
                linewidth=linewidth,
                markersize=markersize,
                colornorm=tmpcolornorm,
                title=title,
                radPlot=radPlot,
                figsize=figsize,
            )
            fig_dict[short_key] = fig

            # calculate ROC data
            if x_val is not None:
                mu_predictions_val, std_predictions_val = estimates_val[key]
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=next(colorcylcer),
                    variant=bounds_variant_NOMU,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )
    if nomu_dj is not None:
        estimates = nomu_dj.calculate_mean_std(x=x_grid)  # estimate mean and std

        if x_val is not None:
            estimates_val = nomu_dj.calculate_mean_std(x=x_val)

        # plot each model in class instance
        for key, model in nomu_dj.models.items():
            short_key = "NOMU_DJ_{}".format(int(re.findall(r"\d+", key)[0]))
            mu_predictions, std_predictions = estimates[key]
            B = calculate_uncertainty_bounds(
                c=c_NOMU_DJ,
                variant=bounds_variant_NOMU_DJ,
                mu_predictions=mu_predictions,
                std_predictions=std_predictions,
                model_key=key,
            )
            if only_uncertainty:
                y = B["uncertainty"]
                title = "Plot Shows Uncertainty of " + B["label_uncertainty"]
            else:
                y = B["bounds"]["Upper Bound"]
                title = "Plot Shows Upper Bound of " + B["label_bounds"]

            title_details = "\nStatic Parameters:" + ", ".join(
                [k + ":{}".format(v) for k, v in static_parameters.items()]
            )
            title_details += "\nParameters: " + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in nomu_dj.parameters[
                        list(nomu_dj.parameters.keys())[0]
                    ].items()
                ]
            )
            title_details = title_details.replace(", optimizer", "\noptimizer")
            title_details_dict[short_key] = key + title_details

            if show_details_title:
                title += title_details
            fig = fig_2d(
                xPlot[0],
                xPlot[1],
                y.T[0].reshape(resolution, resolution),
                x_train,
                x_aug,
                y_train,
                linewidth=linewidth,
                markersize=markersize,
                colornorm=tmpcolornorm,
                title=title,
                radPlot=radPlot,
                figsize=figsize,
            )
            fig_dict[short_key] = fig

            # calculate ROC data
            if x_val is not None:
                mu_predictions_val, std_predictions_val = estimates_val[key]
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=next(colorcylcer),
                    variant=bounds_variant_NOMU_DJ,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )
    # (iv) plot MC Dropout Models
    if mc_dropout is not None:

        # predict from dropout model
        if x_val is not None:
            predictions = mc_dropout.predict(x=concatenated_data)
        else:
            predictions = mc_dropout.predict(x=x_grid)

        # estimate mean and std
        estimates = mc_dropout.calculate_mean_std(x=None, predictions=predictions)

        # plot each model in class instance
        for key, model in mc_dropout.models.items():
            short_key = "DO_{}".format(int(re.findall(r"\d+", key)[0]))
            predictions_plot = predictions[key]
            if x_val is not None:
                # split predictions:
                predictions_val = [
                    x[-n_val:, :] for x in predictions_plot
                ]  # dropout predictions on validation set -> for ROC
                predictions_plot = [
                    x[0:-n_val, :] for x in predictions_plot
                ]  # CHANGE TO INCLUDE ALSO VALIDATION DATA IN PREDICTION

            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[key],
                plot_indices=np.arange(resolution ** din),
                n_val=n_val if x_val is not None else x_val,
            )
            # bounds
            B = calculate_uncertainty_bounds(
                c=c_DO,
                variant=bounds_variant_DO,
                mu_predictions=mu_predictions,
                std_predictions=std_predictions,
                raw_predictions=predictions_plot,
                model_key=key,
            )
            if only_uncertainty:
                y = B["uncertainty"]
                title = "Plot Shows Uncertainty of " + B["label_uncertainty"]
            else:
                y = B["bounds"]["Upper Bound"]
                title = "Plot shows Upper Bound of " + B["label_bounds"]

            # set title
            title_details = "\nStatic Parameters: " + ", ".join(
                [k + ":{}".format(v) for k, v in static_parameters.items()]
            )
            title_details += "\nParameters: " + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in mc_dropout.parameters[key].items()
                    if k not in dynamic_parameters_DO
                ]
            )
            title_details = title_details.replace(", optimizer", "\noptimizer")
            title_details_dict[short_key] = key + title_details

            if show_details_title:
                title += title_details
            fig = fig_2d(
                xPlot[0],
                xPlot[1],
                y.T[0].reshape(resolution, resolution),
                x_train,
                y_train,
                linewidth=linewidth,
                markersize=markersize,
                colornorm=tmpcolornorm,
                title=title,
                radPlot=radPlot,
                figsize=figsize,
            )
            fig_dict[short_key] = fig

            # calculate ROC data based on validation data
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=key,
                    color=next(colorcylcer),
                    variant=bounds_variant_DO,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    predictions=predictions_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )

    # (vi) plot Gaussian Process (GP) Models
    if gp is not None:

        # estimate mean and std
        if x_val is not None:
            concatenated_data = np.concatenate((x_grid, np.squeeze(x_val)), axis=0)
            estimates = gp.calculate_mean_std(x=concatenated_data)
        else:
            estimates = gp.calculate_mean_std(x=x_grid)

        # plot each model in class instance
        for key, model in gp.models.items():
            short_key = "GP_{}".format(int(re.findall(r"\d+", key)[0]))
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[key],
                plot_indices=np.arange(resolution ** din),
                n_val=n_val if x_val is not None else x_val,
            )

        # bounds
        B = calculate_uncertainty_bounds(
            c=c_GP,
            variant=bounds_variant_GP,
            mu_predictions=mu_predictions,
            std_predictions=std_predictions,
            model_key=key,
        )
        if only_uncertainty:
            y = B["uncertainty"]
            title = "Plot Shows Uncertainty of " + B["label_uncertainty"]
        else:
            y = B["bounds"]["Upper Bound"]
            title = "Plot Shows Upper Bound of " + B["label_bounds"]

        title_details = "\nStatic Parameters: " + ", ".join(
            [k + ":{}".format(v) for k, v in static_parameters.items()]
        )
        title_details += "\nParameters: " + ", ".join(
            [
                k + ":{}".format(v)
                for k, v in gp.parameters[key].items()
                if k not in dynamic_parameters_GP
            ]
        )
        title_details += (
            "\nInitial: {} | Optimum: {} | Log-Marginal-Likelihood: {}".format(
                gp.initial_kernels[key],
                model.kernel_,
                round(model.log_marginal_likelihood(model.kernel_.theta), 3),
            )
        )
        title_details_dict[short_key] = key + title_details

        if show_details_title:
            title += title_details
        fig = fig_2d(
            xPlot[0],
            xPlot[1],
            y.T[0].reshape(resolution, resolution),
            x_train,
            y_train,
            linewidth=linewidth,
            markersize=markersize,
            colornorm=tmpcolornorm,
            title=title,
            radPlot=radPlot,
            figsize=figsize,
        )
        fig_dict[short_key] = fig

        # calculating ROC data on validation set
        if x_val is not None:
            ROC_list, label_list, color_list = update_metrics_list(
                ROC_list=ROC_list,
                label_list=label_list,
                color_list=color_list,
                label=key,
                color=next(colorcylcer),
                variant=bounds_variant_GP,
                m=mu_predictions_val,
                sig=std_predictions_val,
                y_val=y_val,
                captured_flag=captured_flag,
                c_max=c_max_ROC,
                custom_c_grid=custom_c_grid_ROC,
                cp_max=cp_max_ROC,
                resolution=resolution_ROC,
                interpolation=interpolation,
            )

    # (vii) plot Deep Ensemble Models
    if deep_ensemble is not None:

        # estimate mean and std
        if x_val is not None:
            concatenated_data = np.concatenate((x_grid, np.squeeze(x_val)), axis=0)
            estimates = deep_ensemble.calculate_mean_std(x=concatenated_data)
        else:
            estimates = deep_ensemble.calculate_mean_std(x=x_grid)

        # plot each model in class instance
        for ensemble_key, ensemble in deep_ensemble.models.items():
            short_ensemble_key = "DE_{}".format(
                int(re.findall(r"\d+", ensemble_key)[0])
            )
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[ensemble_key],
                plot_indices=np.arange(resolution ** din),
                n_val=n_val if x_val is not None else x_val,
            )

            # bounds
            B = calculate_uncertainty_bounds(
                c=c_DE,
                variant=bounds_variant_DE,
                mu_predictions=mu_predictions,
                std_predictions=std_predictions,
                model_key=ensemble_key,
            )
            if only_uncertainty:
                y = B["uncertainty"]
                title = "Plot Shows Uncertainty of " + B["label_uncertainty"]
            else:
                y = B["bounds"]["Upper Bound"]
                title = "Plot Shows Upper Bound of " + B["label_bounds"]

            title_details = "\nStatic Parameters: " + ", ".join(
                [k + ":{}".format(v) for k, v in static_parameters.items()]
            )
            title_details += "\nParameters: " + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in deep_ensemble.parameters[ensemble_key].items()
                    if k not in dynamic_parameters_DE
                ]
            )
            title_details = title_details.replace(", optimizer", "\noptimizer")
            title_details_dict[short_ensemble_key] = ensemble_key + title_details

            if show_details_title:
                title += title_details
            fig = fig_2d(
                xPlot[0],
                xPlot[1],
                y.T[0].reshape(resolution, resolution),
                x_train,
                y_train,
                linewidth=linewidth,
                markersize=markersize,
                colornorm=tmpcolornorm,
                title=title,
                radPlot=radPlot,
                figsize=figsize,
            )
            fig_dict[short_ensemble_key] = fig

            # calculate ROC data based on validation data
            if x_val is not None:
                ROC_list, label_list, color_list = update_metrics_list(
                    ROC_list=ROC_list,
                    label_list=label_list,
                    color_list=color_list,
                    label=ensemble_key,
                    color=next(colorcylcer),
                    variant=bounds_variant_DE,
                    m=mu_predictions_val,
                    sig=std_predictions_val,
                    y_val=y_val,
                    captured_flag=captured_flag,
                    c_max=c_max_ROC,
                    custom_c_grid=custom_c_grid_ROC,
                    cp_max=cp_max_ROC,
                    resolution=resolution_ROC,
                    interpolation=interpolation,
                )

    # (ix) save plot and title details and ROC
    savepath = None
    if save:

        if filepath is not None:

            dt = datetime.now().strftime("%d_%m_%Y_%H-%M-%S")

            for key, fig in fig_dict.items():

                if static_parameters.get("random_locations"):
                    fname = "_seed{}_".format(static_parameters.get("seed")) + dt
                else:
                    fname = "_" + dt

                # save plot for each model
                savepath = os.path.join(filepath, "{}_Plot".format(key) + fname)
                fig.savefig(savepath + ".pdf", format="pdf", transparent=True)

                # save info for each model
                savepath = os.path.join(
                    filepath, "{}_Plot_Info".format(key) + fname + ".txt"
                )
                with open(savepath, "w") as f:
                    f.write(title_details_dict[key])
                f.close()
                plt.close()
            # savepath for ROC
            if static_parameters.get("random_locations"):
                ROC_plot_path = os.path.join(
                    filepath,
                    "Plot_Roc_seed{}_".format(static_parameters.get("seed")) + dt,
                )

            else:
                ROC_plot_path = os.path.join(filepath, "Plot_Roc_" + dt)

        else:
            print("No filepath for saving the UB plot specified.")
            print("No filepath for saving the info file specified.")
            print(
                "No filepath for saving the ROC plot specified."
            ) if x_val is not None else None

    if x_val is not None:
        return_dict = ROC_plot(
            ROC_list,
            label_list,
            color_list,
            logy=logy_ROC,
            linethreshy=linethreshy_ROC,
            savepath=ROC_plot_path if save else None,
            captured_flag=captured_flag,
            bounds_variant_DO=bounds_variant_DO,
            bounds_variant_GP=bounds_variant_GP,
            bounds_variant_DE=bounds_variant_DE,
            bounds_variant_NOMU=bounds_variant_NOMU,
        )
        return return_dict


def plot_sig_width_boxplot(
    std_predictions_val, std_predictions_train, info="", short_key=""
):
    # plt.boxplot(std_predictions_val)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.boxplot([std_predictions_val, std_predictions_train], labels=["val", "train"])
    plt.yscale("log")
    plt.figtext(0.5, 0.95, short_key, fontsize=12, ha="center")
    plt.figtext(0.5, 0.9, info, fontsize=4, ha="center")
    # ax.set_title(short_key)
    # plt.show()


# %%
def fig_2d(
    x: np.array,
    y: np.array,
    z: np.array,
    x_train: np.array,
    y_train: np.array,
    radPlot: int = 1.1,
    markersize: int = 60,
    linewidth: int = 1,
    figsize: Tuple[int, int] = (10, 10),
    plot_type: str = "contour",
    colornorm: Optional[colors.Normalize] = None,
    title: str = "",
) -> plt.figure:

    """Helper function for plot_predictions_2d. Creates 2d 'surface' or 'contour' plots.

    Arguments
    ----------
    x :
        x-coordinates of main plot.
    y :
        y-coordinates of main plot.
    z :
        z-coordinates of main plot, e.g. upper bound  at (x,y) or uncertainty measure at (x,y).
    x_train :
        Input (features) of training points, will be plotted additionally.
    y_train :
        Output (targets) of training points, will be plotted additionally.
    radPlot :
        Box bounds for the plotting area.
    markersize :
        Markersize of markers in plots.
    linewidth :
        Linewidth of lines in plots.
    figsize :
        Size of figure, e.g. (16,9).
    plot_type :
        Type for 2d plot: 'contour'.
    colornorm :
        Colornorm of contour plot, used for plotting the additional info like x_train
        in the correct color.
    title :
        Title of plot.

    Returns
    -------
    fig :
        Created matplotlib.pyplot figure object.

    """

    fig = plt.figure(figsize=figsize)
    plt.xlim(-radPlot, radPlot)
    plt.ylim(-radPlot, radPlot)

    if plot_type == "contour":
        # plot contour
        plt.contourf(
            x, y, z, locator=ticker.LinearLocator(numticks=100), norm=colornorm
        )

        # plot datapoints
        trs = mlines.Line2D(
            [],
            [],
            markeredgecolor="black",
            markerfacecolor="w",
            marker="o",
            linestyle="None",
            markersize=10,
            label="Input Training Points",
            linewidth=linewidth,
        )
        plt.scatter(
            x_train[:, 0],
            x_train[:, 1],
            marker="o",
            s=markersize,
            c=y_train,
            norm=colornorm,
            edgecolors="w",
            linewidth=linewidth,
        )
        handles = [trs]

        # (iv b) finish plot
        cb = plt.colorbar(ticks=np.arange(0, 2.5, 0.5))
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(25)
        # To specify the number of ticks on both or any single axes
        plt.locator_params(axis="y", nbins=5)
        plt.locator_params(axis="x", nbins=5)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(handles=handles, shadow=True, fontsize=25)
        plt.scatter(
            x_train[:, 0],
            x_train[:, 1],
            marker="o",
            s=markersize,
            c="w",
            norm=colornorm,
            edgecolors="w",
            linewidth=linewidth,
        )
        plt.xlabel(r"x$_1$", fontsize=25)
        plt.ylabel(r"x$_2$", fontsize=25, rotation=0)

        plt.grid(True)
        plt.title(title, fontsize="small")
        plt.tight_layout()

    return fig


# %%
def plot_irradiance(
    x_train: np.array,
    y_train: np.array,
    x_aug: np.array,
    y_aug: np.array,
    x_val: Optional[np.array],
    y_val: Optional[np.array],
    filepath: Optional[str],
    captured_flag: bool,
    static_parameters: List[str],
    # NOMU
    nomu: Optional[NOMU],
    dynamic_parameters_NOMU: List[str],
    bounds_variant_NOMU: str,
    c_NOMU: float,
    # NOMU_DJ
    nomu_dj: Optional[NOMU_DJ],
    dynamic_parameters_NOMU_DJ: List[str],
    bounds_variant_NOMU_DJ: str,
    c_NOMU_DJ: float,
    # GP
    gp: Optional[GaussianProcess],
    dynamic_parameters_GP: List[str],
    bounds_variant_GP: str,
    c_GP: float,
    # DO
    mc_dropout: Optional[McDropout],
    dynamic_parameters_DO: List[str],
    sample_size_DO: int,
    bounds_variant_DO: str,
    c_DO: float,
    # DE
    deep_ensemble: Optional[DeepEnsemble],
    dynamic_parameters_DE: List[str],
    bounds_variant_DE: str,
    c_DE: float,
    #
    radPlot: float = 1.01,
    save: bool = False,
    markersize: int = 6,
    transparency: float = 0.2,
    linewidth: int = 2,
    resolution: int = 1920,
    extrapolate: bool = False,
    interpolation: str = "linear",
    show_details_title: bool = False,
) -> Union[NoReturn, Dict[str, Tuple[float, float, float]]]:

    """Plots the bounds and Roc-curves for specified models (NOMU,GP,DO,BNN,DE).

    Arguments
    ----------
    x_train :
        Input (features) of training points.
    y_train :
        Output (targets) of training points.
    x_aug :
        Input (features) of augmented points.
    y_aug :
        Output (targets) of augmented points.
    x_val :
        Input (features) of validation points.
    y_val :
        Output (targets) of validation points.
    filepath :
        Absolute Path for saving the plots.
    captured_flag :
        For calculation of coverage probability in ROC. Ifset to True, only connsiders by the bounds captured points, i.e.
        points x where LB(x)<=x<=UB(x).
    static_parameters :
        Static parameters across different models for the plot title and info.
    nomu :
        Instance of NOMU class.
    dynamic_parameters_NOMU :
        Dynamic parameters for NOMU models for plot title and info.
    bounds_variant_NOMU :
        Variant for calculation of uncertainty bounds. For NOMU one has the options 'standard': [y+/-c_NOMU*r] or
        'normal': [y+/-normal_quantile[c_NOMU]*r] on how the credible intervals are computed.
    c_NOMU :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU)/2), where Phi denotes the cdf of N(0,1).
    nomu_dj :
        Instance of NOMU_DJ class.
    dynamic_parameters_NOMU_DJ :
        Dynamic parameters for NOMU_DJ models for plot title and info.
    bounds_variant_NOMU_DJ :
        Variant for calculation of uncertainty bounds. For NOMU_DJ one has the options 'standard': [y+/-c_NOMU_DJ*r] or
        'normal': [y+/-normal_quantile[c_NOMU_DJ]*r] on how the credible intervals are computed.
    c_NOMU_DJ :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_NOMU_DJ='standard' this is interpreted as a scalar and
        when bounds_variant_NOMU_DJ='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_NOMU_DJ] is
        given for the lower/upper bound as Phi^-1((1-/+c_NOMU_DJ)/2), where Phi denotes the cdf of N(0,1).
    mc_dropout :
        Instance of McDropout class.
    dynamic_parameters_DO :
        Dynamic parameters for McDropout models for plot title and info.
    sample_size_DO :
        Number of samples drawn to estimate mean and std on a given input point x.
    bounds_variant_DO :
        Variant for calculation of uncertainty bounds. For McDropout one has the options 'standard': [mean+/-c_DO*std] or
        'sample': [lower_sample_quantile[c_DO], upper_sample_quantile[c_DO]] on how the credible intervals are computed,
        i.e, lower_sample_quantile[c_GP]/upper_sample_quantile[c_DO] is given F_emp^-1((1-/+c_DO)/2), where F_emp denotes the empirical cdf of a set of samples.
    c_DO :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DO='standard' this is interpreted as a scalar and
        when bounds_variant_DO='sample' this is intepreted as the coverage probability of the (sample) credible interval.
    gp :
        Instance of GaussianProcess class.
    dynamic_parameters_GP :
        Dynamic parameters for GaussianProcess models for plot title and info.
    bounds_variant_GP :
        Variant for calculation of uncertainty bounds. For GaussianProcess one has the options 'standard': [mean+/-c_GP*std] or
        'normal': [mean+/-normal_quantile[c_GP]*std] on how the credible intervals are computed.
    c_GP :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_GP='standard' this is interpreted as a scalar and
        when bounds_variant_GP='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_GP] is
        given for the lower/upper bound as Phi^-1((1-/+c_GP)/2), where Phi denotes the cdf of N(0,1).
    deep_ensemble :
        Instance of DeepEnsemble class.
    dynamic_parameters_DE :
        Dynamic parameters for DeepEnsemble models for plot title and info.
    bounds_variant_DE :
        Variant for calculation of uncertainty bounds. For DeepEnsemble one has the options 'standard': [mean+/-c_DE*std] or
        'normal': [mean+/-normal_quantile[c_DE]*std] on how the credible intervals are computed.
    c_DE :
        Scaling factor for the calculation of the uncertainty bounds. When bounds_variant_DE='standard' this is interpreted as a scalar and
        when bounds_variant_DE='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c_DE] is
        given for the lower/upper bound as Phi^-1((1-/+c_DE)/2), where Phi denotes the cdf of N(0,1).
    radPlot :
        Box bounds for the plotting area.
    save :
        Bool for saving the plots in the specified filepath.
    markersize :
        Markersize of markers in plots.
    linewidth :
        Linewidth of lines in plots.
    extrapolate :
        Bool for extrapolating by radPlot from the x_train samples. (assumes x_train to be ordered).
    interpolation :
        Interpolation type for calculating the np.quantile on samples.
    show_details_title :
        Bool for showing details in the plot title.

    Returns (optional)
    ------------------
    return_dict :
         Dict consisting of AUC's, Max std-factor's/prob's and threshhold factor's/prob's in dict for each model.

    """

    if extrapolate:
        start = x_train[0, 0]
        end = x_train[-1, 0] + radPlot
    else:
        start = -radPlot
        end = radPlot
    # prepare for calculation of ROC
    if x_val is not None:
        n_val = x_val.shape[0]

    # if factors are not provided as lists, change to list
    if not isinstance(c_NOMU, list):
        c_NOMU = [c_NOMU]
    if not isinstance(c_NOMU_DJ, list):
        c_NOMU_DJ = [c_NOMU_DJ]
    if not isinstance(c_DO, list):
        c_DO = [c_DO]
    if not isinstance(c_GP, list):
        c_GP = [c_GP]
    if not isinstance(c_DE, list):
        c_DE = [c_DE]

    # (iii) plot NN uncertainty bounds
    if nomu is not None:
        title, title_details, xPlot, sorted_index, concatenated_data = set_up_plot(
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            markersize,
            start,
            end,
            static_parameters,
            resolution,
        )
        title_details += "UB Neural Network Parameters\n " + ", ".join(
            [
                k + ":{}".format(v)
                for k, v in nomu.parameters[list(nomu.parameters.keys())[0]].items()
                if k not in dynamic_parameters_NOMU
            ]
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        # estimate mean and std
        if x_val is not None:
            estimates = nomu.calculate_mean_std(x=concatenated_data)
        else:
            estimates = nomu.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        col = 0
        for key, model in nomu.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )

            # plot mean predictions
            plot_nnub = plt.plot(
                xPlot,
                mu_predictions,
                linewidth=linewidth,
                label="NOMU "
                + str(col + 1)
                + r": $\hat{f}$"
                + ", ".join(
                    [""]
                    + [
                        k + ": " + str(nomu.parameters[key][k])
                        for k in dynamic_parameters_NOMU
                    ]
                ),
                linestyle="-",
                color="C" + str(col),
            )

            # plot bounds AND uncertainty
            for c in c_NOMU:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_NOMU,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                plt.plot(
                    xPlot,
                    B["uncertainty"],
                    linewidth=linewidth,
                    label="NOMU "
                    + str(col + 1)
                    + ": "
                    + B["label_uncertainty"]
                    + ", ".join(
                        [""]
                        + [
                            k + ": " + str(nomu.parameters[key][k])
                            for k in dynamic_parameters_NOMU
                        ]
                    ),
                    linestyle=":",
                    color=plot_nnub[0].get_color(),
                )

                fill_nnub = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], (B["bounds"]["Upper Bound"])[::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_nnub[0].get_color(),
                    ec="None",
                    label="NOMU " + str(col + 1) + ": " + B["label_bounds"],
                )
            finish_and_save_plot(
                title,
                title_details,
                show_details_title,
                save,
                filepath,
                static_parameters,
                model_name="NOMU",
                labels=["NOMU: " + r" $\hat{f}\pm$" + B["label_bounds"]],
                ltys=[(plot_nnub[0], fill_nnub[0])],
            )
            col += 1

    if nomu_dj is not None:
        title, title_details, xPlot, sorted_index, concatenated_data = set_up_plot(
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            markersize,
            start,
            end,
            static_parameters,
            resolution,
        )
        title_details += "NOMU DJ Parameters\n " + ", ".join(
            [
                k + ":{}".format(v)
                for k, v in nomu_dj.parameters[
                    list(nomu_dj.parameters.keys())[0]
                ].items()
                if k not in dynamic_parameters_NOMU_DJ
            ]
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        # estimate mean and std
        if x_val is not None:
            estimates = nomu_dj.calculate_mean_std(x=concatenated_data)
        else:
            estimates = nomu_dj.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        col = 0
        for key, model in nomu_dj.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )

            # plot mean predictions
            plot_nnub_dj = plt.plot(
                xPlot,
                mu_predictions,
                linewidth=linewidth,
                label="NOMU DJ "
                + str(col + 1)
                + r": $\hat{f}$"
                + ", ".join(
                    [""]
                    + [
                        k + ": " + str(nomu_dj.parameters[key][k])
                        for k in dynamic_parameters_NOMU_DJ
                    ]
                ),
                linestyle="-",
                color="C" + str(col),
            )

            # plot bounds AND uncertainty
            for c in c_NOMU_DJ:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_NOMU_DJ,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                plt.plot(
                    xPlot,
                    B["uncertainty"],
                    linewidth=linewidth,
                    label="NOMU DJ "
                    + str(col + 1)
                    + ": "
                    + B["label_uncertainty"]
                    + ", ".join(
                        [""]
                        + [
                            k + ": " + str(nomu_dj.parameters[key][k])
                            for k in dynamic_parameters_NOMU_DJ
                        ]
                    ),
                    linestyle=":",
                    color=plot_nnub_dj[0].get_color(),
                )

                fill_nnub_dj = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], (B["bounds"]["Upper Bound"])[::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_nnub_dj[0].get_color(),
                    ec="None",
                    label="NOMU DJ " + str(col + 1) + ": " + B["label_bounds"],
                )
            finish_and_save_plot(
                title,
                title_details,
                show_details_title,
                save,
                filepath,
                static_parameters,
                model_name="NOMU_DJ",
                labels=["NOMU DJ: " + r" $\hat{f}\pm$" + B["label_bounds"]],
                ltys=[(plot_nnub_dj[0], fill_nnub_dj[0])],
            )
            col += 1
    # (iv) plot Gaussian Process (GP) models
    if gp is not None:
        title, title_details, xPlot, sorted_index, concatenated_data = set_up_plot(
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            markersize,
            start,
            end,
            static_parameters,
            resolution,
        )
        title_details = title_details + "\nGaussian Process Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in gp.parameters[list(gp.parameters.keys())[0]].items()
                    if (k not in dynamic_parameters_GP and v is not None)
                ]
            )
            + "\n"
        )

        # estimate mean and std
        if x_val is not None:
            estimates = gp.calculate_mean_std(x=concatenated_data)
        else:
            estimates = gp.calculate_mean_std(x=xPlot)

        # plot each model in class instance
        col = 1
        for key, model in gp.models.items():
            title_details = (
                title_details
                + key
                + ":  Initial: {} | Optimum: {} | Log-Marginal-Likelihood: {}".format(
                    gp.initial_kernels[key],
                    model.kernel_,
                    round(model.log_marginal_likelihood(model.kernel_.theta), 4),
                )
            )

            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates=estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot mean predictions
            plot_gpr = plt.plot(
                xPlot,
                mu_predictions,
                linewidth=linewidth,
                label="GP"
                + r": $\hat{f}$",  # + ', '.join([''] + [k + ': ' + str(gp.parameters[key][k]) for k in dynamic_parameters_GP]),
                linestyle="-",
                color="C" + str(col),
            )
            # plot bounds
            for c in c_GP:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_GP,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                fill_gpr = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_gpr[0].get_color(),
                    ec="None",
                    label="GP: " + B["label_bounds"],
                )
            finish_and_save_plot(
                title,
                title_details,
                show_details_title,
                save,
                filepath,
                static_parameters,
                model_name="GP",
                labels=["GP: " + r" $\hat{f}\pm$" + B["label_bounds"]],
                ltys=[(plot_gpr[0], fill_gpr[0])],
            )
            col += 1

    # (v) plot MC Dropout Models
    if mc_dropout is not None:
        title, title_details, xPlot, sorted_index, concatenated_data = set_up_plot(
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            markersize,
            start,
            end,
            static_parameters,
            resolution,
        )
        # predict from dropout model
        if x_val is not None:
            predictions = mc_dropout.predict(x=concatenated_data)
        else:
            predictions = mc_dropout.predict(x=xPlot)

        # estimate mean and std
        estimates = mc_dropout.calculate_mean_std(x=None, predictions=predictions)

        # plot each model in class instance
        col = 2
        for key, model in mc_dropout.models.items():
            predictions_plot = predictions[key]

            if x_val is not None:
                # split predictions
                predictions_plot = [
                    x[sorted_index] for x in predictions_plot
                ]  # all dropout predictions sorted -> for plot

            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot mean prediction
            plot_dp = plt.plot(
                xPlot,
                mu_predictions,
                linewidth=linewidth,
                label="MCDO"
                + r": $\hat{f}$",  # + ', '.join([''] + [k + ': ' + str(mc_dropout.parameters[key][k]) for k in dynamic_parameters_DO]),
                color="C" + str(col),
            )
            # plot bounds
            for c in c_DO:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_DO,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    raw_predictions=predictions_plot,
                    model_key=None,
                )

                fill_do = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_dp[0].get_color(),
                    ec="None",
                    label="MC DO: "
                    + B["label_bounds"]
                    + ", ".join(
                        [""]
                        + [
                            k + ": " + str(mc_dropout.parameters[key][k])
                            for k in dynamic_parameters_DO
                        ]
                    ),
                )
            col += 1

        title_details = title_details + "\nMc Dropout Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in mc_dropout.parameters[
                        list(mc_dropout.parameters.keys())[0]
                    ].items()
                    if k not in dynamic_parameters_DO
                ]
            )
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")
        finish_and_save_plot(
            title,
            title_details,
            show_details_title,
            save,
            filepath,
            static_parameters,
            model_name="MCDO",
            labels=["MCDO: " + r" $\hat{f}\pm$" + B["label_bounds"]],
            ltys=[(plot_dp[0], fill_do[0])],
        )

    # (vii) plot Deep Ensemble Models
    if deep_ensemble is not None:
        title, title_details, xPlot, sorted_index, concatenated_data = set_up_plot(
            x_train,
            y_train,
            x_aug,
            y_aug,
            x_val,
            y_val,
            markersize,
            start,
            end,
            static_parameters,
            resolution,
        )
        # estimate mean and std
        if x_val is not None:
            estimates = deep_ensemble.calculate_mean_std(x=concatenated_data)
        else:
            estimates = deep_ensemble.estimate_models(x=xPlot)

        # plot
        col = 3
        for ensemble_key, ensemble in deep_ensemble.models.items():
            (
                mu_predictions,
                std_predictions,
                mu_predictions_val,
                std_predictions_val,
            ) = split_mu_sigma(
                estimates[ensemble_key],
                plot_indices=sorted_index,
                n_val=n_val if x_val is not None else x_val,
            )
            # plot the mean output
            plot_de = plt.plot(
                xPlot,
                mu_predictions,
                linewidth=linewidth,
                label="DE"
                + r": $\hatf$",  # + ', '.join([''] + [k + ': ' + str(deep_ensemble.parameters[ensemble_key][k]) for k in dynamic_parameters_DE]),
                color="C" + str(col),
            )
            # plot the bounds
            for c in c_DE:
                B = calculate_uncertainty_bounds(
                    c=c,
                    variant=bounds_variant_DE,
                    mu_predictions=mu_predictions,
                    std_predictions=std_predictions,
                    model_key=None,
                )

                fill_de = plt.fill(
                    np.concatenate([xPlot.reshape(-1, 1), xPlot.reshape(-1, 1)[::-1]]),
                    np.concatenate(
                        [B["bounds"]["Lower Bound"], B["bounds"]["Upper Bound"][::-1]]
                    ),
                    alpha=transparency,
                    fc=plot_de[0].get_color(),
                    ec="None",
                    label="DE: " + B["label_bounds"],
                )  # + ', '.join([''] + [k + ': ' + str(deep_ensemble.parameters[ensemble_key][k]) for k in dynamic_parameters_DE]))
            col += 1

        title_details = title_details + "\nDeep Ensemble Parameters:\n"
        title_details = (
            title_details
            + " "
            + ", ".join(
                [
                    k + ":{}".format(v)
                    for k, v in deep_ensemble.parameters[
                        list(deep_ensemble.parameters.keys())[0]
                    ].items()
                    if k not in dynamic_parameters_DE
                ]
            )
        )
        title_details = title_details.replace(", optimizer", "\noptimizer")

        finish_and_save_plot(
            title,
            title_details,
            show_details_title,
            save,
            filepath,
            static_parameters,
            model_name="DE",
            labels=["DE: " + r" $\hat{f}\pm$" + B["label_bounds"]],
            ltys=[(plot_de[0], fill_de[0])],
        )


# %%
def ROC_plot(
    ROC_list: List[np.array],
    label_list: List[str],
    color_list: List[str],
    logy: bool = False,
    savepath: Optional[str] = None,
    linethreshy: float = 0.1,
    captured_flag: bool = True,
    bounds_variant_DO: str = "standard",
    bounds_variant_DO2: str = "standard",
    bounds_variant_BNN: str = "standard",
    bounds_variant_GP: str = "standard",
    bounds_variant_DE: str = "standard",
    bounds_variant_NOMU: str = "standard",
) -> Dict[str, Tuple[float, float, float]]:

    """Creates Roc plot of all considered models.

    Arguments
    ----------
    ROC_list :
        List of return values of custom_ROC (i.e., np.array of shape
        (#of c values required for full coverage, 5) with
        columns: |coverage probability|mean width|mean log width|nlpd|c-value|).
    label_list :
        List of labels for each model for plot title and legend.
    color_list :
        List of colors for each models given by the return value of plt.plot()[0].get_color(),
        i.e., a string defining the color.
    logy :
        Bool for a log scale of the oc plot.
    savepath :
        Absolute Path for saving the ROC plot..
    linethreshy :
        Factor that determines at which point on the y-axis the scaling should
        switch from linear to log for the Roc plot..
    captured_flag :
        For calculation of coverage probability in ROC. Ifset to True, only connsiders by the bounds captured points, i.e.
        points x where LB(x)<=x<=UB(x).
    bounds_variant_DO :
        Variant for calculation of uncertainty bounds. For DO one has the options 'standard': [mean+/-c_DO*std] or
        'sample': [lower_sample_quantile[c_DO], upper_sample_quantile[c_DO]] on how the credible intervals are computed,
        i.e, lower_sample_quantile[c_DO]/upper_sample_quantile[c_DO] is given F_emp^-1((1-/+c_DO)/2), where F_emp denotes the empirical cdf of a set of samples.
    bounds_variant_DO2 :
        Variant for calculation of uncertainty bounds. For DO2 one has the options 'standard': [mean+/-c_DO*std].
    bounds_variant_BNN :
        Variant for calculation of uncertainty bounds. For BNN one has the options 'standard': [mean+/-c_BNN*std] or
        'sample': [lower_sample_quantile[c_BNN], upper_sample_quantile[c_DO]] on how the credible intervals are computed,
        i.e, lower_sample_quantile[c_BNN]/upper_sample_quantile[c_BNN] is given F_emp^-1((1-/+c_BNN)/2), where F_emp denotes the empirical cdf of a set of samples.
    bounds_variant_GP :
        Variant for calculation of uncertainty bounds. For GaussianProcess one has the options 'standard': [mean+/-c_GP*std] or
        'normal': [mean+/-normal_quantile[c_GP]*std] on how the credible intervals are computed.
    bounds_variant_DE :
        Variant for calculation of uncertainty bounds. For DeepEnsemble one has the options 'standard': [mean+/-c_DE*std] or
        'normal': [mean+/-normal_quantile[c_DE]*std] on how the credible intervals are computed.
     bounds_variant_NOMU :
        Variant for calculation of uncertainty bounds. For NOMU one has the options 'standard': [y+/-c_NOMU*r] or
        'normal': [y+/-normal_quantile[c_NOMU]*r] on how the credible intervals are computed.
    Returns
    -------
        return_dict :
            Dict of 3 (one for each of MW, MlogW and NLPD) dicts consisting
            of AUC's, Max std-factor's/prob's for each model.

    """

    # Return AUC's, Max Max std-factor's/prob's and threshhold factor's/prob's in dict for each model
    return_dict = {}

    for j, y in enumerate(["MW", "MlogW", "NLPD"]):
        plt.figure()
        plt.xlabel("CP")
        return_dict_metric = {}
        y_label = y
        if captured_flag and y in ["MW", "MlogW"]:
            y_label += " captured"

        title = ""

        # variable to capture maximum tick on y-axis
        maxroc = 1

        for i, roc in enumerate(ROC_list):
            # determine row for which MlogW/NLPD capture first validation point
            inf_row = 0
            if y in ["NLPD"]:
                while np.isinf(roc[inf_row, 1 + j]):
                    inf_row += 1
                roc = roc[inf_row:, :]

            # 1) plot roc
            plt.plot(roc[:, 0], roc[:, 1 + j], label=label_list[i], color=color_list[i])
            # update maximum y value
            maxroc = max([maxroc, np.max(roc[:, 1 + j])])

            # 2) Calculate and return AUC's and Max c / minimal NLPD and corresponding cp
            # in dict for each model

            if y == "NLPD":
                min_nll = np.min(roc[:, 3])
                min_ind = np.argmin(roc[:, 3])
                min_c = roc[min_ind, -1]
                min_cp = roc[min_ind, 0]
                return_dict_metric[label_list[i]] = (min_nll, min_cp, min_c)

                # write minimal nlpd c value into plot
                plt.text(
                    x=roc[min_ind, 0] * 0.97,
                    y=min_nll * 1.1,
                    s="{:.2f}".format(min_c),
                    fontsize=7,
                )

                # write maximal c value into plot
                plt.text(
                    x=roc[-1, 0] + 0.001,
                    y=roc[-1, 1 + j] - 0.001,
                    s="{:.2f}".format(roc[-1, -1]),
                    fontsize=7,
                )

                # set title
                if label_list[i][0] == "M" and bounds_variant_DO in [
                    "sample",
                    "normal",
                ]:
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_DO
                        + " variant)\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                        )
                    )
                elif label_list[i][0] == "M" and bounds_variant_DO2 in ["sample"]:
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_DO2
                        + " variant)\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                        )
                    )
                elif label_list[i][0] == "B" and bounds_variant_BNN in [
                    "sample",
                    "normal",
                ]:
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_BNN
                        + " variant)\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                        )
                    )
                elif label_list[i][0] == "G" and bounds_variant_GP == "normal":
                    title += label_list[
                        i
                    ] + "\n AUC: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                        *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                    )
                elif label_list[i][0] == "U" and bounds_variant_NOMU == "normal":
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_NOMU
                        + " variant)\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                        )
                    )
                elif label_list[i][0] == "D" and bounds_variant_DE == "normal":
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_DE
                        + " variant)\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                        )
                    )
                else:
                    title += label_list[
                        i
                    ] + "\n min NLPD: {:.5f}, min NLPD factor: {:.2f}, min NLPD cp: {:.2f}, max cp: {:.2f}, min cp: {:.2f} \n".format(
                        *[min_nll, min_c, min_cp, roc[-1, 0], roc[0, 0]]
                    )
            else:
                auc = np.trapz(roc[:, 1 + j], roc[:, 0])
                return_dict_metric[label_list[i]] = (auc, roc[-1, -1])

                # find c corresponding to logythreshold
                try:
                    linethresh_ind = np.min(
                        np.where(np.greater(roc[:, 1 + j], linethreshy))
                    )
                except:
                    linethresh_ind = -1
                linethresh_c = roc[linethresh_ind, -1]

                # write threshold c value into plot
                plt.text(
                    x=roc[linethresh_ind, 0] * 0.97,
                    y=linethreshy * 1.1,
                    s="{:.2f}".format(linethresh_c),
                    fontsize=7,
                )

                # write maximal c value into plot
                plt.text(
                    x=roc[-1, 0] + 0.001,
                    y=roc[-1, 1 + j] - 0.001,
                    s="{:.2f}".format(roc[-1, -1]),
                    fontsize=7,
                )

                # set title
                if label_list[i][0] == "M" and bounds_variant_DO in [
                    "sample",
                    "normal",
                ]:
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_DO
                        + " variant)\n AUC: {:.5f}, max prob: {:.2f}, thr prob: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[
                                auc,
                                roc[-1, -1],
                                roc[linethresh_ind, -1],
                                roc[-1, 0],
                                roc[0, 0],
                            ]
                        )
                    )
                elif label_list[i][0] == "B" and bounds_variant_BNN in [
                    "sample",
                    "normal",
                ]:
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_BNN
                        + " variant)\n AUC: {:.5f}, max prob: {:.2f}, thr prob: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[
                                auc,
                                roc[-1, -1],
                                roc[linethresh_ind, -1],
                                roc[-1, 0],
                                roc[0, 0],
                            ]
                        )
                    )
                elif label_list[i][0] == "G" and bounds_variant_GP == "normal":
                    title += label_list[
                        i
                    ] + "\n AUC: {:.5f}, max prob: {:.2f}, thr prob: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                        *[
                            auc,
                            roc[-1, -1],
                            roc[linethresh_ind, -1],
                            roc[-1, 0],
                            roc[0, 0],
                        ]
                    )
                elif label_list[i][0] == "U" and bounds_variant_NOMU == "normal":
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_NOMU
                        + " variant)\n AUC: {:.5f}, max prob: {:.2f}, thr prob: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[
                                auc,
                                roc[-1, -1],
                                roc[linethresh_ind, -1],
                                roc[-1, 0],
                                roc[0, 0],
                            ]
                        )
                    )
                elif label_list[i][0] == "D" and bounds_variant_DE == "normal":
                    title += (
                        label_list[i]
                        + " ("
                        + bounds_variant_DE
                        + " variant)\n AUC: {:.5f}, max prob: {:.2f}, thr prob: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                            *[
                                auc,
                                roc[-1, -1],
                                roc[linethresh_ind, -1],
                                roc[-1, 0],
                                roc[0, 0],
                            ]
                        )
                    )
                else:
                    title += label_list[
                        i
                    ] + "\n AUC: {:.5f}, max std factor: {:.2f}, thr std factor: {:.2f}, max cp: {:.2f}, min cp: {:.2f}\n".format(
                        *[
                            auc,
                            roc[-1, -1],
                            roc[linethresh_ind, -1],
                            roc[-1, 0],
                            roc[0, 0],
                        ]
                    )

        plt.legend()
        if logy:
            y_label += f". log-thr={linethreshy}"
            plt.yscale("symlog", linthreshy=linethreshy)

        # plt.yticks(ticks=yticks)
        plt.grid(which="both")
        plt.ylabel(y_label)
        plt.title(title, fontsize=7)
        plt.hlines(y=linethreshy, xmin=0, xmax=1, color="grey", lw=1.5)
        plt.tight_layout()
        if savepath is not None:
            if j == 0:
                plt.savefig(savepath + ".pdf", format="pdf", transparent=True)
            elif j == 1:
                plt.savefig(
                    savepath.replace("Roc", "LogRoc") + ".pdf",
                    format="pdf",
                    transparent=True,
                )
            elif j == 2:
                plt.savefig(
                    savepath.replace("Roc", "NLPD") + ".pdf",
                    format="pdf",
                    transparent=True,
                )
            plt.close()
        # collect metric dicts in return dict
        return_dict[y] = return_dict_metric
    return return_dict


#%%
def set_up_plot(
    x_train,
    y_train,
    x_aug,
    y_aug,
    x_val,
    y_val,
    markersize,
    start,
    end,
    static_parameters,
    resolution,
):

    plt.figure(figsize=(16, 9))
    # To specify the number of ticks on both or any single axes
    plt.locator_params(axis="y", nbins=7)
    plt.locator_params(axis="x", nbins=7)
    plt.ylim = (-2, 2.5)  # (-1.5, 1.5)  #
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.plot(
        x_train[:, :-1],
        y_train,
        "ko",
        markersize=markersize,
        label="Training Data",
        zorder=5,
    )
    # vertical dashed lines
    ind_interpolation_end = np.where((np.diff(x_val[0:100], axis=0) * 10 // 1) > 0)[0]
    ind_interpolation_end = np.concatenate(
        (ind_interpolation_end, ind_interpolation_end[[-1]] + 20)
    )
    ind_interpolation_start = ind_interpolation_end - 19
    dashed_line_locations = np.concatenate(
        (x_val[ind_interpolation_start], x_val[ind_interpolation_end])
    )
    for x in dashed_line_locations:
        plt.axvline(x=x, linestyle="--", color="grey")
    plt.xlim(start, end)
    title = (
        "Static Parameters: "
        + ", ".join([k + ":{}".format(v) for k, v in static_parameters.items()])
        + "\n"
    )
    title_details = ""
    xPlot = np.linspace(start, end, resolution)
    plt.plot(
        x_val, y_val, "r+", markersize=markersize, label="Validation Data", zorder=5
    )
    # create concatenated data
    if x_val is not None:
        concatenated_data = np.concatenate((xPlot, np.squeeze(x_val)))
        sorted_index = np.argsort(concatenated_data)
        xPlot = concatenated_data[sorted_index]  # all x points sorted -> for plot
    return (title, title_details, xPlot, sorted_index, concatenated_data)


# %%
def finish_and_save_plot(
    title,
    title_details,
    show_details_title,
    save,
    filepath,
    static_parameters,
    showlegend=True,
    showtitle=False,
    showgrid=False,
    model_name="",
    labels=None,
    ltys=None,
):
    # (viii) finish plot
    if showlegend:
        if ltys is not None:
            plt.legend(ltys, labels, loc="upper left", shadow=True, fontsize=20)
            plt.locator_params(axis="y", nbins=5)
            plt.locator_params(axis="x", nbins=5)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
        else:
            plt.legend(loc="upper left", shadow=True, fontsize="small")
    if showgrid:
        plt.grid(True)
    if showtitle:
        if show_details_title:
            plt.title(title + title_details, fontsize="small")
        else:
            plt.title(title, fontsize="small")
    plt.tight_layout()

    # (ix) save plot and title details and ROC
    savepath = None
    if save:

        if filepath is not None:

            # save plot
            if static_parameters.get("random_locations"):
                fname = "_seed{}_".format(
                    static_parameters.get("seed")
                ) + datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            else:
                fname = "_" + datetime.now().strftime("%d_%m_%Y_%H-%M-%S")
            savepath = os.path.join(
                filepath, "Plot_Bounds_" + model_name + fname + ".pdf"
            )
            plt.savefig(savepath, format="pdf", transparent=True)

            # save info
            savepath = os.path.join(
                filepath, "Plot_Info_" + model_name + fname + ".txt"
            )
            with open(savepath, "w") as f:
                f.write(title + title_details)
            f.close()
            plt.close()
        else:
            print("No filepath for saving the bounds plot specified.")
            print("No filepath for saving the info file specified.")
