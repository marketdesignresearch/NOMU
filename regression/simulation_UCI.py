#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file is used for the regression experiments on the UCI and UCI gap data sets.

"""

# Libs
import os
import sys

# read UCI data set name, experiment type and gap_dim from console input
# Example call "python3 src/simulation_uci.py UCI-Gap 2 333"
if len(sys.argv) < 3:
    print(
        "Error: Pleas enter the experiment type, the gap dimension and a seed as console input (e.g. UCI-Gap 2 333)."
    )
    exit()

experiment_type = sys.argv[1]
gap_dim = int(sys.argv[2])
seed_set = int(sys.argv[3])
print("################################################################")
print(
    f"Set experiment type to '{experiment_type}', gap dimension to '{gap_dim}' and seed to '{seed_set}'."
)

# ------------------------------------------------------------------------- #
# disable eager execution for tf.__version__ 2.3
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------- #

from collections import OrderedDict
from datetime import datetime
import numpy as np
import pandas as pd
import random

# Own Modules
from algorithms.model_classes.nomu import NOMU as NOMU
from algorithms.model_classes.nomu_dj import NOMU_DJ as NOMU_DJ
from algorithms.util import timediff_d_h_m_s, custom_cgrid
from algorithms.nll_calibration import nll_calibration
from plot_functions.plot_functions import calculate_metrics
from data_generation.data_generator import generate_augmented_data
from data_generation.import_data import uci_data_selector

# %% (0) REPRODUCABILITY
tf.compat.v1.keras.backend.clear_session()

# Global seed
SEED = seed_set

# 1. Set `python` built-in pseudo-random generator at a fixed value
random.seed(SEED)

# 2. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED)

# 3. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(SEED)

# 4. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)
# %% (1) Model parameters

# (i) DATA
#############################################################################################################################

# load data and set fixed params
ds_name = "boston"
df = uci_data_selector(ds_name)

# fixed data params
din = df.shape[1] - 1
n_data = df.shape[0]


n_train = n_data  # number of data points
n_aug = 100  # number of artificial data points

if experiment_type == "UCI":
    function_name = "UCI"
elif experiment_type == "UCI-Gap":
    function_name = "UCI-Gap"
else:
    print("Error: Experiment type input not valid.")
    exit()

if experiment_type == "UCI-Gap" and (gap_dim > din - 1 or gap_dim < 0):
    print(
        f"Error: Chosen gap dimension '{gap_dim}' not in input dimension range 0-{din-1}."
    )
    exit()


# val_size = 0.2 in UCI According to Hernandez-Lobato, Adams 2015 Sec. 5.1

if experiment_type == "UCI":
    val_size = 0.2  # proportion of training data that should be used as validation data
    epochs = 400
    mu_sqr = 0.1  # weight L2-loss for training data points
    mu_exp = 0.01  # weight exp-loss

elif experiment_type == "UCI-Gap":
    val_size = 0.2  # proportion of training data that should be used as validation data
    epochs = 40
    mu_sqr = 0.01  # weight L2-loss for training data points
    mu_exp = 0.1  # weight exp-loss

val_seed = seed_set * 2 + 1
test_seed = seed_set * 3 + 3
print("val seed", val_seed)
print("test seed", test_seed)

nnarchitecture = "small"  #'small' -> layer, 50 n. ; 'large' many layers, more n.
normalize_data = True
aug_in_training_range = True
aug_range_epsilon = 0.05

calibrate_NOMUs = (
    True  # if true, r_tansform for NOMU & NOMU_DJ is calibrated on validation data
)
Refit_NOMU = (
    True  # if true, model are refit on training and validation data after calibration
)

# (ii) NOMU
#############################################################################################################################

# (a) model parameters
# ----------------------------------------------------------------------------------------------------------------------------
####################
FitNOMU = True

stable_loss = 2  # for the new stable_loss use stable_loss=2
c_sqr_stable_aug_loss = (
    1 if stable_loss == 1 else None
)  # parameter for squared term of more stable loss
c_negativ_stable = 1
c_huber_stable = 1
c_exp = 30  # constant in exp loss

####################
if nnarchitecture == "large":
    layers = (
        din,
        2 ** 10,
        2 ** 10,
        2 ** 10,
        1,
    )  # layers incl. input and output
elif nnarchitecture == "small":
    layers = (din, 50, 1)  # layers incl. input and output

batch_size = min(n_train, n_aug)
l2reg = 1e-9  # L2-regularization on weights of main architecture
l2reg_sig = 1e-4  # L2-regularization on weights of side architecture
# l2reg_sig = 1e-8  # L2-regularization on weights of side architecture
seed_init = seed_set * 1000
print("seed_init", seed_init)


# (a2) NOMU DISJOINT (inherits all parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
# set to True for training NOMU without backward passes in dashed connections from \hat{f} network to \hat{r}_f network
#####################
FitNOMUDj = True
#####################

# (b) optimizer
# ----------------------------------------------------------------------------------------------------------------------------
optimizer = "Adam"  # select optimizer stochastic gradient descent: 'SGD' or adaptive moment estimation: 'Adam'
clipnorm = 1.0  # set maximal gradient norm or set to None
learning_rate_NOMU = 0.01

# (c) loss parameters
# ----------------------------------------------------------------------------------------------------------------------------
MCaug = True  # Monte Carlo Approximation of the Integrals in the specified Loss with uniform sampling
c_exp = 30  # constant in exp loss

if nnarchitecture == "large":
    side_layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # r-architecture
elif nnarchitecture == "small":
    side_layers = (din, 50, 1)  # r-architecture

r_transform = [
    "custom_min_max"
]  # 'id' ,'relu_cut', or 'custom_min_max' where 'relu_cut' and 'custom_min_max' uses r_min and r_max.
r_min = 1e-1  # minimum r for numerical stability
r_max = 1  # asymptotically maximum r
dropout_prob = None  # dropout probability

# (iii) SAVING & LOADING
#############################################################################################################################
savemodel = True  # save models and parameters?
loadmodel = False  # load models and parameters?
# %% (2) Data parameters
static_parameters = OrderedDict(
    zip(
        ["function_name", "n_train", "n_aug"],
        [function_name, n_train, n_aug],
    )
)

(
    x,
    y,
    x_train,
    y_train,
    x_aug,
    y_aug,
    x_val,
    y_val,
    x_test,
    y_test,
    n_train,
    n_val,
    n_test,
) = generate_augmented_data(
    data=function_name,
    df=df,
    din=din,
    dout=1,
    n_train=n_train,
    plot=True,
    n_aug=n_aug,
    random_aug=MCaug,
    val_size=val_size,
    val_seed=val_seed,
    test_seed=test_seed,
    gap_dim=gap_dim,
    figsize=(10, 10),
    aug_in_training_range=aug_in_training_range,
    aug_range_epsilon=aug_range_epsilon,
)


# %% (3) create save folder and/or enter load path
savepath = None
loadpath = None
if savemodel:
    foldername = "_".join(
        [function_name, ds_name, datetime.now().strftime("%d_%m_%Y_%H-%M-%S")]
    )
    savepath = os.path.join(os.getcwd(), foldername)
    os.mkdir(savepath)  # if folder exists automatically an FileExistsError is thrown
if loadmodel:
    folder = input("Enter folder for loadpath: ")
    loadpath = os.path.join(os.getcwd(), folder)
# %% (4a) Train NOMU with or woithout backward passes in dashed connections
start0 = datetime.now()


if FitNOMU:
    nomu = NOMU()
    if loadmodel:
        nomu.load_models(
            absolutepath=loadpath,
            model_numbers=[
                1
            ],  # give a list of model numbers for loading multiple NOMU models
            verbose=0,
        )
    else:
        nomu.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            optimizer_learning_rate=learning_rate_NOMU,
            seed_init=seed_init,
            MCaug=MCaug,
            n_train=n_train,
            n_aug=n_aug,
            mu_sqr=mu_sqr,
            mu_exp=mu_exp,
            c_exp=c_exp,
            r_transform=r_transform,
            r_min=r_min,
            r_max=r_max,
            l2reg_sig=l2reg_sig,
            side_layers=side_layers,
            stable_loss=stable_loss,
            c_sqr_stable_aug_loss=c_sqr_stable_aug_loss,
            c_negativ_stable=c_negativ_stable,
            c_huber_stable=c_huber_stable,
            optimizer_clipnorm=clipnorm,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu.initialize_models(verbose=0)
        nomu.compile_models(verbose=0)
        nomu.fit_models(x=x, y=y, verbose=1)
        nomu.plot_histories(yscale="log")
    if savemodel:
        nomu.save_models(absolutepath=savepath)

if FitNOMUDj:
    nomu_dj = NOMU_DJ()
    if loadmodel:
        nomu_dj.load_models(
            absolutepath=loadpath,
            model_numbers=[
                1
            ],  # give a list of model numbers for loading multiple NOMU_DJ models
            verbose=0,
        )
    else:
        nomu_dj.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            optimizer_learning_rate=learning_rate_NOMU,
            seed_init=seed_init,
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
            stable_loss=stable_loss,
            c_sqr_stable_aug_loss=c_sqr_stable_aug_loss,
            c_negativ_stable=c_negativ_stable,
            c_huber_stable=c_huber_stable,
            optimizer_clipnorm=clipnorm,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu_dj.initialize_models(verbose=0)
        nomu_dj.compile_models(verbose=0)
        nomu_dj.fit_models(x=x, y=y, verbose=0)
        nomu_dj.plot_histories(yscale="log")
    if savemodel:
        nomu_dj.save_models(absolutepath=savepath)

print("CP Main: Post fit block")

end0 = datetime.now()
print(
    "\nTraining Time Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end0 - start0)),
    "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
)
# %% (6a) Set plot & metric calc parameters

bounds_variant_NOMU = "standard"
bounds_variant_NOMU_DJ = "standard"


captured_flag = False  # if true, calculate mw captured in ROC plot. else, mw.
linethreshy_ROC = 1
cp_max_ROC = 1  # maximum coverage probability as stopping criteria
# linear grid
c_max_ROC = None
resolution_ROC = None

# custom grid
grid_min = 2 * 1e-2
grid_max = 1e2
steps = 200
max_power_of_two = 20
custom_c_grid_ROC = custom_cgrid(
    grid_min=grid_min, grid_max=grid_max, steps=steps, max_power_of_two=max_power_of_two
)

add_nlpd_constant = True  # add constant for nlpd metric?

# %% #####################
# Calibrate NOMU & NOMU_DJ
##########################

y_val = np.array(y_val)  # add bc predict expects np array not pd obj.

# r_min grid
grid_min = 5 * 1e-3
grid_max = 0.8
steps = 100
factor = (grid_max / grid_min) ** (1 / steps)
r_min_grid = np.array([grid_min * factor ** i for i in range(steps)])

# r_max grid
grid_min = 0.8
grid_max = 10
steps = 80
factor = (grid_max / grid_min) ** (1 / steps)
r_max_grid = np.array([grid_min * factor ** i for i in range(steps)])

if calibrate_NOMUs:
    if FitNOMU:
        nll_dict_r_grids = nomu.nll_calibration_r_grids(
            r_min_grid=r_min_grid,
            r_max_grid=r_max_grid,
            c_grid=custom_c_grid_ROC[1:],
            x_val=x_val,
            y_val=y_val,
            add_nlpd_constant=True,
        )
    if FitNOMUDj:
        try:
            nll_dict_r_grids.update(
                nomu_dj.nll_calibration_r_grids(
                    r_min_grid=r_min_grid,
                    r_max_grid=r_max_grid,
                    c_grid=custom_c_grid_ROC[1:],
                    x_val=x_val,
                    y_val=y_val,
                    add_nlpd_constant=True,
                )
            )
        except NameError:
            nll_dict_r_grids = nomu_dj.nll_calibration_r_grids(
                r_min_grid=r_min_grid,
                r_max_grid=r_max_grid,
                c_grid=custom_c_grid_ROC[1:],
                x_val=x_val,
                y_val=y_val,
                add_nlpd_constant=True,
            )

#%%

print("\nCalculate Metrics")
print("**************************************************************************")
results = calculate_metrics(
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
    #
    save_plot=True,
    save_info=False,
    logy_ROC=False,
    linethreshy_ROC=linethreshy_ROC,
    cp_max_ROC=cp_max_ROC,
    c_max_ROC=c_max_ROC,
    custom_c_grid_ROC=custom_c_grid_ROC,
    resolution_ROC=resolution_ROC,
    #
    plot_std_boxplot=False,
    x_train=x_train,
    #
    add_nlpd_constant=add_nlpd_constant,
)


# %%
rows = list(results[list(results.keys())[0]].keys())
cols = list(results.keys())[1:] + ["100%C", "argminCP", "argminC"]
metrics = pd.DataFrame([[0] * len(cols)] * len(rows), index=rows, columns=cols)
for k, v in results.items():
    for (
        k2,
        v2,
    ) in v.items():
        if k == "MW":
            metrics.loc[k2, k] = np.round(v2[0], 3)
            metrics.loc[k2, "100%C"] = np.round(v2[1], 3)
        if k == "NLPD":
            metrics.loc[k2, k] = np.round(v2[0], 3)
            metrics.loc[k2, "argminCP"] = np.round(v2[1], 3)
            metrics.loc[k2, "argminC"] = np.round(v2[2], 3)
        if k == "MlogW":
            metrics.loc[k2, k] = np.round(v2[0], 3)
print()
print(metrics)
print()

NLPD_grid = pd.DataFrame(None, index=custom_c_grid_ROC, columns=rows)
for k, v in results["NLPD_grid"].items():
    NLPD_grid.loc[:, k] = np.round(v, 3)
print(NLPD_grid)
print()


#%% Refit on combination of training and validation dataset


if Refit_NOMU and FitNOMU:
    print("Refit NOMU on train & validation")

    r_min_nomu = []
    r_max_nomu = []
    for key in nomu.model_keys:
        r_min_nomu.append(nomu.parameters[key]["r_min"])
        r_max_nomu.append(nomu.parameters[key]["r_max"])
    print(r_min_nomu)
    print(r_max_nomu)

    x_trainval = np.concatenate([x_val, np.zeros((x_val.shape[0], 1))], axis=1)
    x_trainval = np.concatenate([x_trainval, x], axis=0)
    y_trainval = np.concatenate([np.array(y_val).reshape(-1, 1), y], axis=0)
    nomu = NOMU()
    if loadmodel:
        nomu.load_models(absolutepath=loadpath, model_numbers=[1, 2], verbose=0)
    else:
        nomu.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            optimizer_learning_rate=learning_rate_NOMU,
            seed_init=seed_init,
            MCaug=MCaug,
            n_train=n_train,
            n_aug=n_aug,
            mu_sqr=mu_sqr,
            mu_exp=mu_exp,
            dropout_prob=dropout_prob,
            c_exp=c_exp,
            r_transform=r_transform,
            r_min=r_min_nomu,
            r_max=r_max_nomu,
            l2reg_sig=l2reg_sig,
            side_layers=side_layers,
            stable_loss=stable_loss,
            c_sqr_stable_aug_loss=c_sqr_stable_aug_loss,
            c_negativ_stable=c_negativ_stable,
            c_huber_stable=c_huber_stable,
            optimizer_clipnorm=clipnorm,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu.initialize_models(verbose=0)
        nomu.compile_models(verbose=0)
        nomu.fit_models(x=x_trainval, y=y_trainval, verbose=0)
        # nomu.plot_histories(yscale="log")
    if savemodel:
        nomu.save_models(absolutepath=savepath)

if Refit_NOMU and FitNOMUDj:
    print("Refit NOMUDJ on train & validation")

    r_min_nomudj = []
    r_max_nomudj = []
    for key in nomu_dj.model_keys:
        r_min_nomudj.append(nomu_dj.parameters[key]["r_min"])
        r_max_nomudj.append(nomu_dj.parameters[key]["r_max"])
    print(r_min_nomudj)
    print(r_max_nomudj)

    x_trainval = np.concatenate([x_val, np.zeros((x_val.shape[0], 1))], axis=1)
    x_trainval = np.concatenate([x_trainval, x], axis=0)
    y_trainval = np.concatenate([np.array(y_val).reshape(-1, 1), y], axis=0)
    nomu_dj = NOMU_DJ()
    if loadmodel:
        nomu_dj.load_models(absolutepath=loadpath, model_numbers=[1, 2, 3], verbose=0)
    else:
        nomu_dj.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
            optimizer_learning_rate=learning_rate_NOMU,
            seed_init=seed_init,
            MCaug=MCaug,
            n_train=n_train,
            n_aug=n_aug,
            mu_sqr=mu_sqr,
            mu_exp=mu_exp,
            dropout_prob=dropout_prob,
            c_exp=c_exp,
            r_transform=r_transform,
            r_min=r_min_nomudj,
            r_max=r_max_nomudj,
            l2reg_sig=l2reg_sig,
            side_layers=side_layers,
            stable_loss=stable_loss,
            c_sqr_stable_aug_loss=c_sqr_stable_aug_loss,
            c_negativ_stable=c_negativ_stable,
            c_huber_stable=c_huber_stable,
            optimizer_clipnorm=clipnorm,
            normalize_data=normalize_data,
            aug_in_training_range=aug_in_training_range,
            aug_range_epsilon=aug_range_epsilon,
        )
        nomu_dj.initialize_models(verbose=0)
        nomu_dj.compile_models(verbose=0)
        nomu_dj.fit_models(x=x, y=y, verbose=0)
        # nomu_dj.plot_histories(yscale="log")
    if savemodel:
        nomu_dj.save_models(absolutepath=savepath)

#%% Analysis on Test set:

y_test = np.array(y_test)  # nll_calibration expects np array not pd obj.
c_values = metrics["argminC"]  # extract c-values for which NLPD is minimal


cols = ["testNLL", "testC"]
test_nll = pd.DataFrame(
    [[0] * len(cols)] * len(metrics.index), index=metrics.index, columns=cols
)

if FitNOMU:
    for key in nomu.model_keys:
        model_nll = nll_calibration(
            model=nomu,
            x_val=x_test,
            y_val=y_test,
            c_grid=[c_values[key]],
            add_nlpd_constant=True,
        )
        test_nll.loc[key, "testNLL"] = model_nll[key]["nll_min"]
        test_nll.loc[key, "testC"] = model_nll[key]["c_min"]
if FitNOMUDj:
    for key in nomu_dj.model_keys:
        model_nll = nll_calibration(
            model=nomu_dj,
            x_val=x_test,
            y_val=y_test,
            c_grid=[c_values[key]],
            add_nlpd_constant=True,
        )
        test_nll.loc[key, "testNLL"] = model_nll[key]["nll_min"]
        test_nll.loc[key, "testC"] = model_nll[key]["c_min"]

# %%

print("calibrated NLL on val data:")
print(metrics[["NLPD", "argminC"]])
print()
print()

print("NLL on test data:")
print(test_nll[["testNLL", "testC"]])
print()
print()

if FitNOMU:
    for key in nomu.model_keys:
        print("r_min nomu", nomu.parameters[key]["r_min"])
        print("r_max nomu", nomu.parameters[key]["r_max"])
if FitNOMUDj:
    for key in nomu_dj.model_keys:
        print("r_min nomudj", nomu_dj.parameters[key]["r_min"])
        print("r_max nomudj", nomu_dj.parameters[key]["r_max"])
