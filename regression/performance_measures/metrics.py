# -*- coding: utf-8 -*-
"""

This file contains custom ROC creators.

"""
# Libs
import numpy as np
from typing import Union, List, Tuple, Optional, Iterable, Dict
from scipy.stats import norm


# Own Modules
from performance_measures.measures import CP, MW, NLPD

# %%
def metric_curves(
    m: np.array,
    sig: np.array,
    y_val: np.array,
    variant: str = "standard",
    predictions: Optional[List[np.array]] = None,
    interpolation: str = "higher",
    resolution: float = 0.1,
    c_max: float = 1000,
    custom_c_grid: Optional[Iterable[float]] = None,
    cp_max: float = 1,
    log_shift: Optional[float] = 0.001,
    captured_flag: bool = True,
    add_nlpd_constant: bool = False,
) -> np.array:

    """Returns list of metrics for a range of c-values.

    Arguments
    ----------
    m :
        np.array of mean predictions (per validation data point).
    sig :
        np.array of sig predictions (per validation data point).
    y_val :
        np.array of ouptput (target) values of validation data.
    d :
        Dimension of input.
    variant :
        Variant for calculation of uncertainty bounds,
        i.e.,  'standard','normal' and 'sample'.
    predictions :
        List of np.arrays of raw predictions for a specific modeltype
        (used in case method=='sample').
    interpolation :
        Method of interpolation used for sample quantiles if method=='sample',
        i.e., one of 'higher', 'linear', etc.
    resolution :
        Resolution of grid of c-values considered in the metric curve.
    c_max :
        Maximal c-value considered in the metric curve in case variant=='standard'.
        (to be precise: maximal c-value is c_max-resolution)
    custom_c_grid :
        Custom grid of c values which should be iterated. If this is set resolution and c_max is ignored.
    cp_max :
        Maximal coverage probability considered in the metric curve.
    captured_flag :
        For calculation of coverage probability in metric curve. If set to True,
        only connsiders validation points captured by the bounds, i.e.
        points (x,y) where LB(x)<=y<=UB(x).
    log_shift :
        scalar shift for MlogW: the metric will give mean(log(x+log_shift))
        for an array x of widths.
    add_nlpd_constant:
        bool, should constant of Gaussian NLL be added?

    Return
    ----------
    ret :
        np.array of shape (#of c values required for full coverage, 5)
        with columns: |coverage probability|mean width|mean log width|negative log likelihood|c-value|.

    """
    # initialize c, cp
    cp = 0
    CP_list = []
    MW_list = []
    MlogW_list = []
    NLL_list = []
    c_list = []

    if custom_c_grid is None:
        if variant == "standard":
            custom_c_grid = np.arange(0, c_max, resolution)
        else:
            custom_c_grid = np.arange(0, 1, resolution)

    for c in custom_c_grid:
        if variant == "standard":
            nlpd = NLPD(m, sig, y_val, c, add_constant=add_nlpd_constant)
        else:
            # no NLPD for variant = sample or normal currently
            nlpd = None
        NLL_list.append(nlpd)

        if cp < cp_max:
            # probabilistic
            # bounds:       [lower_sample_quantile[c], upper_sample_quantile[c]]
            # uncertainty:  upper_sample_quantile[c] - sample_quantile[0.5]
            if variant == "sample":
                lower = np.quantile(
                    predictions,
                    q=(1 - np.round(c, 10)) / 2,
                    axis=0,
                    interpolation=interpolation,
                )
                upper = np.quantile(
                    predictions,
                    q=(1 + np.round(c, 10)) / 2,
                    axis=0,
                    interpolation=interpolation,
                )
            # probabilistic
            # bounds:       [mu +/- normal_quantile[c]*std]
            # uncertainty:  normal_quantile[c]*std
            elif variant == "normal":
                lower = m + norm.ppf((1 - c) / 2) * sig
                upper = m + norm.ppf((1 + c) / 2) * sig
            # non- probabilistic
            # bounds:       [mu +/- c*std]
            # uncertainty:  c*std
            elif variant == "standard":
                lower = m - c * sig
                upper = m + c * sig

            cp = CP(y_val, lower, upper)
            if captured_flag:
                mw = MW(lower, upper, y=y_val)
                mlw = MW(lower, upper, y=y_val, log=True, log_shift=log_shift)
            else:
                mw = MW(lower, upper)
                mlw = MW(lower, upper, log=True, log_shift=log_shift)

            CP_list.append(cp)
            MW_list.append(mw)
            MlogW_list.append(mlw)
            c_list.append(c)

    ret = np.array((CP_list, MW_list, MlogW_list, NLL_list[: len(c_list)], c_list))
    ret_nlpd = np.asarray(NLL_list)
    return ret.T, ret_nlpd.T


# %%
def update_metrics_list(
    ROC_list: List[np.array],
    NLPD_list: List[np.array],
    label_list: List[str],
    color_list: List[str],
    label: str,
    color: str,
    variant: str,
    m: np.array,
    sig: np.array,
    y_val: Optional[np.array] = None,
    predictions: Optional[List[np.array]] = None,
    interpolation: str = "higher",
    resolution: float = 0.01,
    c_max: int = 100,
    custom_c_grid: Optional[Iterable[float]] = None,
    cp_max: float = 1,
    captured_flag: bool = True,
    log_shift: Optional[float] = 0.001,
    add_nlpd_constant: bool = False,
) -> Tuple[List[List[np.array]], List[str], List[str]]:

    """Creates Roc plot of all considered models.

    Arguments
    ----------
    ROC_list :
        List of return values of custom_ROC (i.e., np.array of shape
        (#of c values required for full coverage, 3) with
        columns: |coverage probability|mean width|mean log width|c-value|).
    NLPD_list :
        List of nlpd values on a custom c_grid (i.e., np.array of shape
        (#of c values in c_grid, ) with
        nll values).
    label_list :
        List of labels for each model for plot title and legend.
    color_list :
        List of colors for each models given by the return value of plt.plot()[0].get_color(),
        i.e., a string defining the color.
    label :
        Label of current model added to label_list.
    color :
        Color of current model added to label_list.
    variant :
        Variant for calculation of uncertainty bounds,
        i.e.,  'standard','normal' and 'sample'.
    m :
        np.array of mean predictions for a specific modeltype.
    sig :
         np.array of std predictions for a specific modeltype.
    y_val :
        Input (targets) of validation points.
    predictions :
        List of np.arrays of raw predictions for a specific modeltype
        (used in case method=='sample').
    interpolation :
        Method of interpolation used for sample quantiles if method=='sample',
        i.e., one of 'higher', 'linear', etc.
    resolution :
        Resolution of grid of c-values considered in the ROC.
    c_max :
        Maximal c-value considered in the ROC in case variant=='standard'.
    cp_max:
        Maximal coverage probability considered in the ROC.
    captured_flag :
        For calculation of coverage probability in ROC. Ifset to True,
        only connsiders validation points captured by the bounds, i.e.
        points (x,y) where LB(x)<=y<=UB(x).
    log_shift :
        scalar shift for MlogW: the metric will give mean(log(x+log_shift))
        for an array x of widths.
    add_nlpd_constant:
        bool, should constant of Gaussian NLL be added?
    Returns
    ----------
    (ROC_list, NLPD_list, label_list, color_list):
        Tuple containing extended lists of ROC-arrays, NLPD-arrays, labels and colors.
    """

    # non-probabilistic ROC: bounds are mu+-c*std for grid of c values
    if variant in ["standard", "normal", "sample"]:
        ROC, NLPD = metric_curves(
            m=m,
            sig=sig,
            y_val=y_val,
            predictions=predictions,
            interpolation=interpolation,
            variant=variant,
            c_max=c_max,
            custom_c_grid=custom_c_grid,
            cp_max=cp_max,
            log_shift=log_shift,
            resolution=resolution,
            captured_flag=captured_flag,
            add_nlpd_constant=add_nlpd_constant,
        )
    else:
        raise NotImplementedError("Variant {} is not implmented yet".format(variant))

    ROC_list.append(ROC)
    NLPD_list.append(NLPD)
    label_list.append(label)
    color_list.append(color)
    return (ROC_list, NLPD_list, label_list, color_list)


# %%
def calculate_uncertainty_bounds(
    c: float,
    variant: str,
    mu_predictions: np.array = None,
    std_predictions: np.array = None,
    raw_predictions: np.array = None,
    model_key: str = None,
    interpolation: str = "linear",
) -> Dict[str, Union[np.array, str]]:

    """Calculates uncertainty bounds and uncertainty measure for either raw_predictions or
    already obtained std_predictions & mu_predictions.

         Arguments
         ----------
         c :
             Scaling factor for the calculation of the uncertainty bounds. When variant='standard' this is interpreted as a scalar,
             when variant='normal' this is intepreted as the coverage probability of the credible interval, i.e, normal_quantile[c] is
             given for the lower/upper bound as Phi^-1((1-/+c)/2), where Phi denotes the cdf of N(0,1), and when variant='sample' this
             is intepreted as the coverage probability of the (sample) credible interval.
         variant :
             Variant for calculation of uncertainty bounds, i.e.,  'standard','normal' and 'sample'.
         mu_predictions :
             np.array of mean predictions for a specific modeltype.
         std_predictions :
             np.array of std predictions for a specific modeltype.
         raw_predictions :
             np.array of raw predictions for a specific modeltype.
         model_key :
             Model key of repsective class instance from NOMU, DO, GP, BNN, DE.
         interpolation :
             Interpolation type for calculating the np.quantile on samples.

    """

    # bounds:       [mu +/- c*std]
    # uncertainty:  c*std
    if variant == "standard":
        label_bounds = (
            r"{}: {}$\cdot\hat\sigma_f$".format(model_key, c)
            if model_key is not None
            else r"{}$\cdot\hat\sigma_f$ ".format(c)
        )
        label_uncertainty = (
            r"{}: {}$\cdot\hat\sigma_f$".format(model_key, c)
            if model_key is not None
            else r"{}$\cdot\hat\sigma_f$".format(c)
        )
        bounds = {
            "Lower Bound": mu_predictions - c * std_predictions,
            "Upper Bound": mu_predictions + c * std_predictions,
        }
        uncertainty = std_predictions * c

    # bounds:       [mu +/- normal_quantile[c]*std]
    # uncertainty:  normal_quantile[c]*std
    elif variant == "normal":
        if c < 0 or c > 1:
            raise ValueError("For variant:normal, c must be given in [0,1].")
        label_bounds = (
            "{}: {}%-normal".format(model_key, int(c * 100))
            if model_key is not None
            else "{}%-normal".format(c * 100)
        )
        label_uncertainty = (
            r"{}: {}$\cdot\hat\sigma_f$".format(
                model_key, round(norm.ppf((1 + c) / 2), 2)
            )
            if model_key is not None
            else r"{}$\cdot\hat\sigma_f$".format(round(norm.ppf((1 + c) / 2), 2))
        )
        bounds = {
            "Lower Bound": mu_predictions + norm.ppf((1 - c) / 2) * std_predictions,
            "Upper Bound": mu_predictions + norm.ppf((1 + c) / 2) * std_predictions,
        }
        uncertainty = norm.ppf((1 + c) / 2) * std_predictions

    # bounds:       [lower_sample_quantile[c], upper_sample_quantile[c]]
    # uncertainty:  upper_sample_quantile[c] - sample_quantile[0.5]
    elif variant == "sample":
        if c < 0 or c > 1:
            raise ValueError("For variant:sample, c must be given in [0,1].")
        label_bounds = (
            "{}: {}%-sample".format(model_key, int(c * 100))
            if model_key is not None
            else "{}%-sample".format(c * 100)
        )
        label_uncertainty = (
            r"{}: sampleQ({})-sampleQ(0.5)".format(model_key, c)
            if model_key is not None
            else r"sampleQ({})-sampleQ(0.5)".format(c)
        )
        bounds = {
            "Lower Bound": np.quantile(
                raw_predictions, q=(1 - c) / 2, axis=0, interpolation=interpolation
            ),
            "Upper Bound": np.quantile(
                raw_predictions, q=(1 + c) / 2, axis=0, interpolation=interpolation
            ),
        }
        uncertainty = bounds["Upper Bound"] - np.quantile(
            raw_predictions, q=0.5, axis=0, interpolation=interpolation
        )
    else:
        raise NotImplementedError(
            "Variant {} for model {} not implemented yet".format(variant, model_key)
        )

    return {
        "label_bounds": label_bounds,
        "bounds": bounds,
        "label_uncertainty": label_uncertainty,
        "uncertainty": uncertainty,
    }


# %%
def split_mu_sigma(
    estimates: Tuple[np.array, np.array],
    plot_indices: np.array,
    n_val: Optional[int] = None,
) -> Tuple[np.array, np.array, Optional[np.array], Optional[np.array]]:

    """Split estimates consisting of mean and std predictions in indices that
    are relevant for plotting and for the calculation of the ROC.

         Arguments
         ----------
         estimates :
             A tuple containing predictions for the mean and stds for a specific model of a specific model
             type.
         plot_indices :
             Index that specifies the relevant part of the estimates  for plotting
         n_val :
             Number of validation points to identify the relevant part of the estimates for the ROC
             calculation.

    """

    mu_predictions, std_predictions = estimates

    if n_val is not None:
        # split predictions
        mu_predictions_val = mu_predictions[
            -n_val:, :
        ]  # mu prediction on validation set -> for ROC
        mu_predictions = mu_predictions[
            plot_indices, :
        ]  # all mu predictions sorted -> for plot
        std_predictions_val = std_predictions[
            -n_val:, :
        ]  # std prediction on validation set -> for ROC
        std_predictions = std_predictions[
            plot_indices, :
        ]  # all std predictions sorted -> for plot
        return (
            mu_predictions,
            std_predictions,
            mu_predictions_val,
            std_predictions_val,
        )
    else:
        return (mu_predictions, std_predictions, None, None)
