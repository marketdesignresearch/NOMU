# -*- coding: utf-8 -*-
"""

This file is used for the regression experiments on the generative testbed 1D-20D functions.

"""
# %%
# Libs
import os
from collections import OrderedDict
from datetime import datetime
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from math import floor
import pandas as pd

pd.set_option("display.max_columns", 20)
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # disable eager execution for tf.__version__ 2.3

# Own Modules
from algorithms.model_classes.nomu import NOMU as NOMU
from algorithms.model_classes.nomu_dj import NOMU_DJ as NOMU_DJ
from algorithms.model_classes.deep_ensemble import DeepEnsemble
from algorithms.model_classes.gaussian_process import GaussianProcess
from algorithms.model_classes.hyper_deep_ensemble import HyperDeepEnsemble
from algorithms.model_classes.mc_dropout import McDropout


from algorithms.util import custom_cgrid, timediff_d_h_m_s
from plot_functions.plot_functions import (
    calculate_metrics,
    plot_predictions,
    plot_predictions_2d,
)
from data_generation.data_generator import generate_augmented_data
from data_generation.data_gen import (
    function_library,
    x_data_gen_Uniform_f,
)
from performance_measures.metric_grid_analysis import metric_grid_analysis
from performance_measures.scores import gaussian_nll_score, mse_score

# %% (1) ALGORITHM and DATA parameters

# (i) GROUND TRUTH FUNCTION
#############################################################################################################################

din = 1  # input dimension
function_name = "GaussianBNN"  # ground truth function
if din == 1:
    prior_std = 0.114
elif din == 2:
    prior_std = 0.102
elif din == 5:
    prior_std = 0.092
elif din == 10:
    prior_std = 0.084
elif din == 20:
    prior_std = 0.078
layers_GaussianBNN = (din, 2 ** 10, 2 ** 11, 2 ** 10, 1)

# (ii) DATA
#############################################################################################################################

n_train = 8 * din  # number of training points
n_aug = 100 * din  # number of augmented points
# For equidistant sampling: n_train, n_aug should be given in powers of din
# (i.e., 2**(x*din) datapoints --> 2**x resolution in each dimension)

n_val = 100 * din  # number of test points.
c_aug = 5  # upper bound for artificial data points

noise_scale = 0
random_locations = True
normalize_data = False
aug_in_training_range = False
aug_range_epsilon = 0.05

# CHOOSE INPUT DATA GENERATOR: either

# UNIFORM:
x_data_gen = x_data_gen_Uniform_f(x_min=-1, x_max=1)
# GAUSSIAN-Manifold:
# =============================================================================
# x_data_gen = x_data_gen_Gauss_f(strongDimensions=5,
#                                 strongEigenValue=0.15,
#                                 weakEigenValue=0.001,
#                                 mean=None,
#                                 EigenValuesVector=None)
# =============================================================================


static_parameters = OrderedDict(
    zip(
        [
            "function_name",
            "dim",
            "n_train",
            "n_aug",
            "c_aug",
            "noise_scale",
            "random_locations",
        ],
        [
            function_name,
            din,
            n_train,
            n_aug,
            c_aug,
            noise_scale,
            random_locations,
        ],
    )
)


# (iii) NOMU
#############################################################################################################################

# (a) model parameters
# ----------------------------------------------------------------------------------------------------------------------------
####################
FitNOMU = True
####################
layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # layers incl. input and output
epochs = 2 ** 10
batch_size = n_train
l2reg = 1e-8  # L2-regularization on weights of main architecture
l2reg_sig = l2reg  # L2-regularization on weights of side architecture
seed_init = None  # Seed for weight initialization

# (a2) OUR MODEL DISJOINT (inherits all parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitNOMUDj = False
#####################

# (b) optimizer
# ----------------------------------------------------------------------------------------------------------------------------
optimizer = "Adam"  # select optimizer stochastic gradient descent: 'SGD' or adaptive moment estimation: 'Adam'

# (c) loss parameters
# ----------------------------------------------------------------------------------------------------------------------------
MCaug = True  # Monte Carlo Approximation of the Integrals in the specified Loss with uniform sampling?
mu_sqr = 0.1  # weight L2-loss for training data points
mu_exp = 0.01  # weight exp-loss
c_exp = 30  # constant in exp loss
side_layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # r-architecture
r_transform = "custom_min_max"
r_min = 0.1  # minimum r for numerical stability
r_max = 1  # asymptotically maximum r
dropout_prob = 0  # dropout probability


# (iv) BENCHMARKS:
#############################################################################################################################

# (a) MC Dropout (inherits some parameters from OUR Model)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitMcDropout = True  # compare to dropout model(s)?
#####################
layers_DO = (din, 2 ** 10, 2 ** 11, 2 ** 10, 1)
epochs_DO = epochs
batch_size_DO = batch_size
seed_init_DO = seed_init
optimizer_DO = "Adam"
loss_DO = "mse"
dropout_prob_DO = 0.2  # probability of dropout for entire model
l2reg_DO = l2reg / n_train * (1 - dropout_prob_DO)  # L2-reg for DO model


# (b) Gaussian Process Regression
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitGP = True  # compare to GP model?
#####################
kernel = "rbf"
whitekernel = False
constant_value = 1  # multiplicative constant factor
constant_value_bounds = (
    1e-5,
    100000,
)  #  bounds for hyperparameter optimization of multiplicative constant factor
length_scale = 1  # length scale of rbf kernel
length_scale_bounds = (
    1e-5,
    1e5,
)  # bounds for hyperparameter optimization of length scale of rbf kernel
noise_level = (
    1e-7 if whitekernel else None
)  # noise level for PIs (only in combination with whitekernel_GP)
noise_level_bounds = (
    (1e-10, 1e-5) if whitekernel else None
)  # bounds for hyperparameter optimization of noise level for PIs (only in combination with whitekernel_GP)
alpha = 1e-7 if not whitekernel else 0  # noise level for CIs
n_restarts_optimizer = 10  # number of restarts for hyperparameter optimization
std_min = 1e-3  # minimal predictive standard deviation

# (c) Deep Ensembles (inherits some parameters from OUR Model)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitDE = True  # Compare to Deep Ensemble
#####################
layers_DE = (din, 256, 1024, 512, 1)
epochs_DE = epochs
batch_size_DE = batch_size
l2reg_DE = l2reg / n_train  # L2-reg for deep ensembles
seed_init_DE = seed_init
optimizer_DE = "Adam"
loss_DE = "mse"  # 'nll' or 'mse' currently (mse generates ensemble with only one mean output and does not learn data noise)
number_of_networks_DE = 5
softplus_min_var_DE = (
    1e-6  # minimum variance for numerical stability (only used if loss == nll)
)

# (e) Hyper Deep Ensembles (inherits some parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitHDE = True  # Compare to Deep Ensemble
#####################
layers_HDE = (din, 256, 1024, 512, 1)
epochs_HDE = epochs
batch_size_HDE = batch_size
test_size_HDE = 0.2  # validation set to calculate the score
n_train_HDE = floor(
    n_train * (1 - test_size_HDE)
)  # number of training points to build hyper deep ensemble
l2reg_HDE = (l2reg / 10 ** 3, l2reg * 10 ** 3)  # log uniform bounds
dropout_prob_HDE = (1e-3, 0.4)  # log uniform bounds
seed_init_HDE = 1
optimizer_HDE = "Adam"
loss_HDE = "mse"  # 'nll' or 'mse' currently (mse generates ensemble with only one mean output and does not learn data noise)
K_HDE = 5
kappa_HDE = 50
stratify_HDE = True  # paper=True
fixed_row_init_HDE = True  # paper=True
refit_HDE = False
softplus_min_var_HDE = (
    1e-6  # minimum variance for numerical stability (only used if loss == nll)
)

# (vi) SAVING & LOADING
#############################################################################################################################

##########################
save_pkl = True
save_histories = True
save_metric_plot = True
save_metric_info = True
plot_bounds = True
##########################

if save_pkl:
    foldername = "_".join(
        ["Multiple_Seeds", function_name, datetime.now().strftime("%d_%m_%Y_%H-%M-%S")]
    )
    savepath = os.path.join(os.getcwd(), foldername)
    os.mkdir(savepath)
else:
    savepath = ""

# %% (2)  Set Instances
number_of_instances = 1  # (set to 200 to reproduce experiments from paper)
my_start_seed = 501  # (set to 501 to reproduce experiments from paper)
seeds = [i for i in range(my_start_seed, number_of_instances + my_start_seed)]

# %% (3) Bounds parameters
c_NOMU = 1
c_NOMU_DJ = 1
c_DO = 4
c_GP = 1
c_DE = 15
c_HDE = 10

bounds_variant_NOMU = "standard"
bounds_variant_NOMU_DJ = "standard"
bounds_variant_DO = "standard"
bounds_variant_GP = "standard"
bounds_variant_DE = "standard"
bounds_variant_HDE = "standard"

# (3a) parameters for ROC-like curves
linethreshy_ROC = 1
c_max_ROC = None
resolution_ROC = None
# custom c-grid
grid_min = 1e-3
grid_max = 1e3
steps = int(1e5)
max_power_of_two = 20
custom_c_grid_ROC = custom_cgrid(
    grid_min=grid_min, grid_max=grid_max, steps=steps, max_power_of_two=max_power_of_two
)

cp_max_ROC = 1  # maximum coverage probability as stopping criteria
captured_flag = False  # if true, calculate mw captured in ROC plot else, mw.
add_nlpd_constant = False  # add constant for nlpd metric?

# %% (4) Run Simulation
start0 = datetime.now()
#
verbose = 0
results = {}
instance = 1


for seed in seeds:
    # REPRODUCABILITY
    # ------------------------------------------------------------------------------------------------------
    tf.compat.v1.keras.backend.clear_session()

    # 1. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 2. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 3. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)

    # 4. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    tf.compat.v1.keras.backend.set_session(sess)
    # ------------------------------------------------------------------------------------------------------

    start1 = datetime.now()
    static_parameters["seed"] = seed
    print(
        "\nInstance {}/{}: generate data for seed {}".format(instance, len(seeds), seed)
    )
    print("**************************************************************************")

    if function_name == "GaussianBNN":
        SEED_BNN = seed * (len(layers_GaussianBNN) - 1) * 2
        f_true = function_library(
            function_name,
            p={
                "layers": layers_GaussianBNN,
                "seed": SEED_BNN,
                "scaled": False,
                "resolution": None,
                "prior_std": prior_std,
                "scaling": "differential_evolution",
                "verbose": 1,
            },
        )

    else:
        f_true = function_library(function_name)

    (x, y, x_train, y_train, x_aug, y_aug, x_val, y_val,) = generate_augmented_data(
        din=din,
        dout=1,
        n_train=n_train,
        n_val=n_val,
        f_true=f_true,
        random=random_locations,
        noise_scale=noise_scale,
        seed=seed,
        plot=True,
        c_aug=c_aug,
        n_aug=n_aug,
        random_aug=MCaug,
        noise_on_validation=0,
        figsize=(10, 10),
        x_data_gen=x_data_gen,
        aug_in_training_range=aug_in_training_range,
        aug_range_epsilon=aug_range_epsilon,
    )

    print("\nFit selected models")
    print("**************************************************************************")
    if FitNOMU:
        nomu = NOMU()
        nomu.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            seed_init=seed,
            MCaug=MCaug,
            n_train=n_train,
            n_aug=n_aug,
            mu_sqr=mu_sqr,
            mu_exp=mu_exp,
            dropout_prob=dropout_prob,
            c_exp=c_exp,
            r_transform=r_transform,
            r_min=r_min,
            r_max=r_max,
            l2reg_sig=l2reg_sig,
            side_layers=side_layers,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu.initialize_models(verbose=verbose)
        nomu.compile_models(verbose=verbose)
        nomu.fit_models(x=x, y=y, verbose=verbose)
        if save_histories:
            nomu.plot_histories(
                yscale="log",
                save_only=True,
                absolutepath=os.path.join(
                    savepath,
                    "Plot_History_seed{}_".format(seed)
                    + start1.strftime("%d_%m_%Y_%H-%M-%S"),
                ),
            )

    if FitNOMUDj:
        nomu_dj = NOMU_DJ()
        nomu_dj.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            seed_init=seed,
            MCaug=MCaug,
            n_train=n_train,
            n_aug=n_aug,
            mu_sqr=mu_sqr,
            mu_exp=mu_exp,
            dropout_prob=dropout_prob,
            c_exp=c_exp,
            r_transform=r_transform,
            r_min=r_min,
            r_max=r_max,
            l2reg_sig=l2reg_sig,
            side_layers=side_layers,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu_dj.initialize_models(verbose=verbose)
        nomu_dj.compile_models(verbose=verbose)
        nomu_dj.fit_models(x=x, y=y, verbose=verbose)
        if save_histories:
            nomu_dj.plot_histories(
                yscale="log",
                save_only=True,
                absolutepath=os.path.join(
                    savepath,
                    "Plot_History_seed{}_".format(seed)
                    + start1.strftime("%d_%m_%Y_%H-%M-%S"),
                ),
            )

    if FitMcDropout:
        mc_dropout = McDropout()
        mc_dropout.set_parameters(
            layers=layers_DO,
            epochs=epochs_DO,
            batch_size=batch_size_DO,
            l2reg=l2reg_DO,
            optimizer_name=optimizer_DO,
            seed_init=seed,
            loss=loss_DO,
            dropout_prob=dropout_prob_DO,
            normalize_data=normalize_data,
        )
        mc_dropout.initialize_models(verbose=verbose)
        mc_dropout.compile_models(verbose=verbose)
        mc_dropout.fit_models(x=x_train[:, :-1], y=y_train, verbose=verbose)
        if save_histories:
            mc_dropout.plot_histories(
                yscale="linear",
                save_only=True,
                absolutepath=os.path.join(
                    savepath,
                    "Plot_History_seed{}_".format(seed)
                    + start1.strftime("%d_%m_%Y_%H-%M-%S"),
                ),
            )

    if FitGP:
        gp = GaussianProcess()
        gp.set_parameters(
            kernel=kernel,
            whitekernel=whitekernel,
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            std_min=std_min,
            normalize_data=normalize_data,
        )
        gp.initialize_models(verbose=0)
        gp.compile_models(verbose=0)
        gp.fit_models(x=x_train[:, :-1], y=y_train, verbose=verbose)

    if FitDE:
        deep_ensemble = DeepEnsemble()
        deep_ensemble.set_parameters(
            layers=layers_DE,
            epochs=epochs_DE,
            batch_size=batch_size_DE,
            l2reg=l2reg_DE,
            optimizer_name=optimizer_DE,
            seed_init=seed,
            loss=loss_DE,
            number_of_networks=number_of_networks_DE,
            softplus_min_var=softplus_min_var_DE,
            normalize_data=normalize_data,
        )
        deep_ensemble.initialize_models(verbose=verbose)
        deep_ensemble.compile_models(verbose=verbose)
        deep_ensemble.fit_models(x=x_train[:, :-1], y=y_train, verbose=verbose)
        if save_histories:
            deep_ensemble.plot_histories(
                yscale="linear",
                save_only=True,
                absolutepath=os.path.join(
                    savepath,
                    "Plot_History_seed{}_".format(seed)
                    + start1.strftime("%d_%m_%Y_%H-%M-%S"),
                ),
            )

    if FitHDE:
        hyper_deep_ensemble = HyperDeepEnsemble()
        hyper_deep_ensemble.set_parameters(
            layers=layers_HDE,
            epochs=epochs_HDE,
            batch_size=batch_size_HDE,
            l2reg=l2reg_HDE,
            optimizer_name=optimizer_HDE,
            seed_init=seed,
            loss=loss_HDE,
            dropout_prob=dropout_prob_HDE,
            K=K_HDE,
            kappa=kappa_HDE,
            test_size=test_size_HDE,
            stratify=stratify_HDE,
            fixed_row_init=fixed_row_init_HDE,
            refit=refit_HDE,
            softplus_min_var=softplus_min_var_HDE,
            normalize_data=normalize_data,
        )
        hyper_deep_ensemble.hyper_deep_ens(
            x=x_train[:, :-1],
            y=y_train,
            score=gaussian_nll_score,
            score_single_model=mse_score,
            random_state=seed,
            verbose=1,
        )
        if save_histories:
            hyper_deep_ensemble.plot_histories(
                yscale="linear",
                save_only=True,
                absolutepath=os.path.join(
                    savepath,
                    "Plot_History_seed{}_".format(seed)
                    + start1.strftime("%d_%m_%Y_%H-%M-%S"),
                ),
            )

    if not plot_bounds or din > 2:
        print("\nCalculate Metrics")
        print(
            "**************************************************************************"
        )
        results[seed] = calculate_metrics(
            x_val=x_val,
            y_val=y_val,
            filepath=savepath,
            captured_flag=captured_flag,
            static_parameters=static_parameters,
            # NOMU
            nomu=nomu if FitNOMU else None,
            dynamic_parameters_NOMU=[],  # CHOOSE
            bounds_variant_NOMU=bounds_variant_NOMU,
            # NOMU_DJ
            nomu_dj=nomu_dj if FitNOMUDj else None,
            dynamic_parameters_NOMU_DJ=[],  # CHOOSE
            bounds_variant_NOMU_DJ=bounds_variant_NOMU_DJ,
            # GP
            gp=gp if FitGP else None,
            dynamic_parameters_GP=[],  # CHOOSE
            bounds_variant_GP=bounds_variant_GP,
            # DO
            mc_dropout=mc_dropout if FitMcDropout else None,
            dynamic_parameters_DO=[],  # CHOOSE
            sample_size_DO=100,  # CHOOSE
            bounds_variant_DO=bounds_variant_DO,
            # DE
            deep_ensemble=deep_ensemble if FitDE else None,
            dynamic_parameters_DE=[],  # CHOOSE
            bounds_variant_DE=bounds_variant_DE,
            # HDE
            hyper_deep_ensemble=hyper_deep_ensemble if FitHDE else None,
            dynamic_parameters_HDE=[],  # CHOOSE
            bounds_variant_HDE=bounds_variant_HDE,
            #
            save_plot=save_metric_plot,
            save_info=save_metric_info,
            logy_ROC=True,
            linethreshy_ROC=linethreshy_ROC,
            cp_max_ROC=cp_max_ROC,
            c_max_ROC=c_max_ROC,
            custom_c_grid_ROC=custom_c_grid_ROC,
            resolution_ROC=resolution_ROC,
            #
            plot_std_boxplot=False,
            x_train=x_train,
            add_nlpd_constant=add_nlpd_constant,
        )

    elif plot_bounds and din == 1:
        print("\nCalculate Metrics & Plot 1D-UBs")
        print(
            "**************************************************************************"
        )
        results[seed] = plot_predictions(
            x_train=x_train,
            y_train=y_train,
            x_aug=x_aug,
            y_aug=y_aug,
            x_val=x_val,
            y_val=y_val,
            f_true=f_true,
            filepath=savepath,
            captured_flag=captured_flag,
            static_parameters=static_parameters,
            # NOMU
            nomu=nomu if FitNOMU else None,
            dynamic_parameters_NOMU=[],  # CHOOSE
            bounds_variant_NOMU=bounds_variant_NOMU,
            c_NOMU=c_NOMU,
            # NOMU_DJ
            nomu_dj=nomu_dj if FitNOMUDj else None,
            dynamic_parameters_NOMU_DJ=[],  # CHOOSE
            bounds_variant_NOMU_DJ=bounds_variant_NOMU_DJ,
            c_NOMU_DJ=c_NOMU_DJ,
            # GP
            gp=gp if FitGP else None,
            dynamic_parameters_GP=[],  # CHOOSE
            bounds_variant_GP=bounds_variant_GP,
            c_GP=c_GP,
            # DO
            mc_dropout=mc_dropout if FitMcDropout else None,
            dynamic_parameters_DO=[],  # CHOOSE
            sample_size_DO=100,  # CHOOSE
            bounds_variant_DO=bounds_variant_DO,
            c_DO=c_DO,
            # DE
            deep_ensemble=deep_ensemble if FitDE else None,
            dynamic_parameters_DE=[],  # CHOOSE
            bounds_variant_DE=bounds_variant_DE,
            c_DE=c_DE,
            # HDE
            hyper_deep_ensemble=hyper_deep_ensemble if FitHDE else None,
            dynamic_parameters_HDE=[],  # CHOOSE
            bounds_variant_HDE=bounds_variant_HDE,
            c_HDE=c_HDE,
            #
            radPlot=1.1,  # CHOOSE
            save=True,  # CHOOSE
            markersize=8,  # CHOOSE
            transparency=0.5,  # CHOOSE
            linewidth=1,  # CHOOSE
            logy_ROC=True,  # CHOOSE
            linethreshy_ROC=linethreshy_ROC,
            c_max_ROC=c_max_ROC,
            custom_c_grid_ROC=custom_c_grid_ROC,
            cp_max_ROC=cp_max_ROC,
            resolution_ROC=resolution_ROC,
            show_details_title=False,  # CHOOSE
            add_nlpd_constant=add_nlpd_constant,
        )

    elif plot_bounds and din == 2:
        print("\nCalculate Metrics & Plot 2D-Uncertainty")
        print(
            "**************************************************************************"
        )
        results[seed] = plot_predictions_2d(
            x_train=x_train,
            y_train=y_train,
            x_aug=x_aug,
            y_aug=y_aug,
            x_val=x_val,
            y_val=y_val,
            f_true=None,
            filepath=savepath,
            captured_flag=captured_flag,
            static_parameters=static_parameters,
            # NOMU
            nomu=nomu if FitNOMU else None,
            dynamic_parameters_NOMU=[],  # CHOOSE
            bounds_variant_NOMU=bounds_variant_NOMU,
            c_NOMU=c_NOMU,
            # NOMU_DJ
            nomu_dj=nomu_dj if FitNOMUDj else None,
            dynamic_parameters_NOMU_DJ=[],  # CHOOSE
            bounds_variant_NOMU_DJ=bounds_variant_NOMU_DJ,
            c_NOMU_DJ=c_NOMU_DJ,
            # GP
            gp=gp if FitGP else None,
            dynamic_parameters_GP=[],  # CHOOSE
            bounds_variant_GP=bounds_variant_GP,
            c_GP=c_GP,
            # DO
            mc_dropout=mc_dropout if FitMcDropout else None,
            dynamic_parameters_DO=[],  # CHOOSE
            sample_size_DO=100,  # CHOOSE
            bounds_variant_DO=bounds_variant_DO,
            c_DO=c_DO,
            # DE
            deep_ensemble=deep_ensemble if FitDE else None,
            dynamic_parameters_DE=[],  # CHOOSE
            bounds_variant_DE=bounds_variant_DE,
            c_DE=c_DE,
            # HDE
            hyper_deep_ensemble=hyper_deep_ensemble if FitHDE else None,
            dynamic_parameters_HDE=[],  # CHOOSE
            bounds_variant_HDE=bounds_variant_HDE,
            c_HDE=c_HDE,
            # 1d also
            radPlot=1.1,  # CHOOSE
            save=True,  # CHOOSE
            resolution=50,  # CHOOSE
            markersize=60,  # CHOOSE
            linewidth=1.5,  # CHOOSE
            logy_ROC=True,  # CHOOSE
            linethreshy_ROC=linethreshy_ROC,
            c_max_ROC=c_max_ROC,
            custom_c_grid_ROC=custom_c_grid_ROC,
            cp_max_ROC=cp_max_ROC,
            resolution_ROC=resolution_ROC,
            show_details_title=False,  # CHOOSE
            # 2d
            colorlimits=[0, 1],  # CHOOSE
            plot_type="contour",  # CHOOSE
            only_uncertainty=True,  # CHOOSE
            add_nlpd_constant=add_nlpd_constant,
        )

    # add c_grid defining parameters
    results[seed]["c_grid"] = {
        "grid_min": grid_min,
        "grid_max": grid_max,
        "steps": steps,
        "max_power_of_two": max_power_of_two,
    }

    instance += 1


# save info
if save_pkl:
    with open(os.path.join(savepath, "results.pkl"), "wb") as f:
        pickle.dump(results, f)
    f.close()

plt.ion()  # turn plot on again
#
end0 = datetime.now()
print(
    "\nTotal Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end0 - start0)),
    "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
)
# %% SUMMARY TO CONSOLE

############################## ENTER
coverage_probability_CI = 0.95
results_per_seed = False
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)
##############################

uQ = (coverage_probability_CI + 1) / 2
QUANTILE = norm.ppf(uQ)
number_of_models = len(results[list(results.keys())[0]]["MW"].keys())

#######
# AUC #
########################################################################################
key = "AUC"
results_auc = {k: {k2: v2[0] for k2, v2 in v["MW"].items()} for k, v in results.items()}
pd_auc = pd.DataFrame.from_dict(results_auc)
pd_auc = pd_auc.T.sort_index()
print("\n\n{}:".format(key))
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_auc)
print()
N = pd_auc.shape[0]
pd_auc_summary = pd_auc.describe()
pd_auc_summary.loc["Mean +/-"] = QUANTILE * pd_auc_summary.loc["std"] / np.sqrt(N)
pd_auc_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
    pd_auc_summary.loc["mean"] + pd_auc_summary.loc["Mean +/-"]
)
pd_auc_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
    pd_auc_summary.loc["mean"] - pd_auc_summary.loc["Mean +/-"]
)
print(pd_auc_summary)
print(
    "#---------------------------------------------------------------------------------"
)
print("\n")
########################################################################################


###########
# AUC-MLW #
########################################################################################
key = "AUC-Mean-Log-Width"
results_auc_log = {
    k: {k2: v2[0] for k2, v2 in v["MlogW"].items()} for k, v in results.items()
}
pd_auc_log = pd.DataFrame.from_dict(results_auc_log)
pd_auc_log = pd_auc_log.T.sort_index()
print("\n\n{}:".format(key))
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_auc_log)
print()
N = pd_auc_log.shape[0]
pd_auc_log_summary = pd_auc_log.describe()
pd_auc_log_summary.loc["Mean +/-"] = (
    QUANTILE * pd_auc_log_summary.loc["std"] / np.sqrt(N)
)
pd_auc_log_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
    pd_auc_log_summary.loc["mean"] + pd_auc_log_summary.loc["Mean +/-"]
)
pd_auc_log_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
    pd_auc_log_summary.loc["mean"] - pd_auc_log_summary.loc["Mean +/-"]
)
print(pd_auc_log_summary)
print(
    "#---------------------------------------------------------------------------------"
)
print("\n")
########################################################################################


################
# AUC ARGMAX C #
########################################################################################
results_auc_maxFactor = {
    k: {k2: v2[1] for k2, v2 in v["MW"].items()} for k, v in results.items()
}
pd_auc_maxFactor = pd.DataFrame.from_dict(results_auc_maxFactor)
pd_auc_maxFactor = pd_auc_maxFactor.T.sort_index()
print("\n\nAUC: ARGMAX C")
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_auc_maxFactor)
print()
print(pd_auc_maxFactor.describe())
print(
    "#---------------------------------------------------------------------------------"
)
print("\n")
########################################################################################


############
# NLL_MIN #
########################################################################################
key = "MIN NLL (MIN NLDP)"
results_nlpd = {
    k: {k2: v2[0] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
}
pd_min_nlpd = pd.DataFrame.from_dict(results_nlpd)
pd_min_nlpd = pd_min_nlpd.T.sort_index()
print("\n\n{}:".format(key))
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_min_nlpd)
print()
N = pd_min_nlpd.shape[0]
pd_min_nlpd_summary = pd_min_nlpd.describe()
pd_min_nlpd_summary.loc["Mean +/-"] = (
    QUANTILE * pd_min_nlpd_summary.loc["std"] / np.sqrt(N)
)
pd_min_nlpd_summary.loc["{}%-CI UB".format(int(coverage_probability_CI * 100))] = (
    pd_min_nlpd_summary.loc["mean"] + pd_min_nlpd_summary.loc["Mean +/-"]
)
pd_min_nlpd_summary.loc["{}%-CI LB".format(int(coverage_probability_CI * 100))] = (
    pd_min_nlpd_summary.loc["mean"] - pd_min_nlpd_summary.loc["Mean +/-"]
)
print(pd_min_nlpd_summary)
print(
    "#---------------------------------------------------------------------------------"
)
print("\n")
########################################################################################


##################
# NLL ARGMIN CP #
########################################################################################
results_nlpd_argmin_cp = {
    k: {k2: v2[1] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
}
pd_nlpd_argmin_cp = pd.DataFrame.from_dict(results_nlpd_argmin_cp)
pd_nlpd_argmin_cp = pd_nlpd_argmin_cp.T.sort_index()
print("\n\nNLL ARGMIN CP:")
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_nlpd_argmin_cp)
print()
print(pd_nlpd_argmin_cp.describe())
print(
    "#---------------------------------------------------------------------------------"
)
print("\n")
########################################################################################


#################
# NLL ARGMIN C #
########################################################################################
results_nlpd_argmin_c = {
    k: {k2: v2[2] for k2, v2 in v["NLPD"].items()} for k, v in results.items()
}
pd_nlpd_argmin_c = pd.DataFrame.from_dict(results_nlpd_argmin_c)
pd_nlpd_argmin_c = pd_nlpd_argmin_c.T.sort_index()
print("\n\nNLL ARGMIN C:")
print(
    "#---------------------------------------------------------------------------------"
)
if results_per_seed:
    print(pd_nlpd_argmin_c)
print()
print(pd_nlpd_argmin_c.describe())
print(
    "#---------------------------------------------------------------------------------"
)
########################################################################################

if "NLPD_grid" in results[list(results.keys())[0]]:
    #############
    # NLL-GRID #
    ########################################################################################
    print("\n\nARGMIN-C for NLL GRID:")
    print(
        "#---------------------------------------------------------------------------------"
    )
    dict_NLPD, min_NLPD_c = metric_grid_analysis(results, metric_key="NLPD_grid")
    results_nlpd_grid_minc = {
        k: dict_NLPD[k].loc[:, min_NLPD_c[k]] for k in dict_NLPD.keys()
    }
    pd_nlpd_grid_minc = pd.DataFrame.from_dict(results_nlpd_grid_minc).sort_index()
    key = "MIN NLL-GRID (d_KL)"
    print("\nNLL distribution for GRID-ARGMIN-C (d_KL):")
    print(
        "#---------------------------------------------------------------------------------"
    )
    if results_per_seed:
        print(pd_nlpd_grid_minc)
    print()
    N = pd_nlpd_grid_minc.shape[0]
    pd_nlpd_grid_minc_summary = pd_nlpd_grid_minc.describe()
    pd_nlpd_grid_minc_summary.loc["Mean +/-"] = (
        QUANTILE * pd_nlpd_grid_minc_summary.loc["std"] / np.sqrt(N)
    )
    pd_nlpd_grid_minc_summary.loc[
        "{}%-CI UB".format(int(coverage_probability_CI * 100))
    ] = (
        pd_nlpd_grid_minc_summary.loc["mean"]
        + pd_nlpd_grid_minc_summary.loc["Mean +/-"]
    )
    pd_nlpd_grid_minc_summary.loc[
        "{}%-CI LB".format(int(coverage_probability_CI * 100))
    ] = (
        pd_nlpd_grid_minc_summary.loc["mean"]
        - pd_nlpd_grid_minc_summary.loc["Mean +/-"]
    )
    print(pd_nlpd_grid_minc_summary)
    print(
        "#---------------------------------------------------------------------------------"
    )
    print("\n")
