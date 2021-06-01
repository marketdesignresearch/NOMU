# -*- coding: utf-8 -*-
"""
This file is used for the regression experiments on the real-data irradiance time series.

"""

# Libs
import os
import random
from collections import OrderedDict
from datetime import datetime
import numpy as np
from math import floor

# ------------------------------------------------------------------------- #
# disable eager execution for tf.__version__ 2.3.0
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------- #

# Own Modules
from algorithms.model_classes.nomu import NOMU
from algorithms.model_classes.nomu_dj import NOMU_DJ
from algorithms.model_classes.mc_dropout import McDropout
from algorithms.model_classes.gaussian_process import GaussianProcess
from algorithms.model_classes.deep_ensemble import DeepEnsemble
from algorithms.model_classes.hyper_deep_ensemble import HyperDeepEnsemble
from plot_functions.plot_functions import plot_irradiance, plot_predictions
from data_generation.data_generator import generate_augmented_data
from algorithms.util import timediff_d_h_m_s
from data_generation.import_data import import_irradiance
from performance_measures.scores import gaussian_nll_score, mse_score

# %% (0) REPRODUCABILITY
tf.compat.v1.keras.backend.clear_session()

# Global seed
SEED = 655

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
df = import_irradiance()
df

# fixed data params
din = 1
n_data = df.shape[0]

df.plot(x="YEAR", y=["11yrCYCLE", "11yrCYCLE+BKGRND"])

# select 'n_strips' (almost) equidistant regions of length 'lenstrips' in the range [start, stop] as validation data
train_size = 2 / 3
start = 20
stop = n_data - 20
n_strips = 5
lenstrips = 20
n_train = n_data - n_strips * lenstrips

n_aug = 2 ** (7)  # number of artificial points.

function_name = "Irradiance"

# (ii) NOMU
#############################################################################################################################

# (a) model parameters
# ----------------------------------------------------------------------------------------------------------------------------
####################
FitNOMU = True
####################
layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # layers incl. input and output
epochs = 2 ** 14
batch_size = n_train
l2reg = 1e-19  # L2-regularization on weights of \hat{f} network
l2reg_sig = l2reg  # L2-regularization on weights of \hat{r}_f network
seed_init = None  # Seed for weight initialization


# (a2) NOMU DISJOINT (inherits all parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
# set to True for training NOMU without backward passes in dashed connections from \hat{f} network to \hat{r}_f network
#####################
FitNOMUDj = False
#####################

# (b) optimizer
# ----------------------------------------------------------------------------------------------------------------------------
optimizer = "Adam"  # select optimizer stochastic gradient descent: 'SGD' or adaptive moment estimation: 'Adam'
clipnorm = None  # set maximal gradient norm or set to None

# (c) loss parameters
# ----------------------------------------------------------------------------------------------------------------------------
MCaug = True  # Monte Carlo Approximation of the Integrals in the specified Loss with uniform sampling?
mu_sqr = 0.1  # weight L2-loss for training data points (pi_sqr from paper)
mu_exp = 0.05  # weight exp-loss (pi_exp from paper)
c_exp = 30  # constant in exp loss
side_layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # r-architecture
r_transform = "custom_min_max"  # either 'id', 'relu_cut' or 'custom_min_max' (latter 2 use r_min and r_max).
r_min = 1e-2  # minimum r
r_max = 2  # asymptotically maximum r

# (iii) BENCHMARKS:
#############################################################################################################################

# (a) MC Dropout (inherits some parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitMcDropout = True  # compare to dropout model(s)?
#####################
layers_DO = (din, 2 ** 10, 2 ** 11, 2 ** 10, 1)
epochs_DO = epochs
batch_size_DO = batch_size
l2reg_DO = l2reg
seed_init_DO = seed_init
optimizer_DO = "Adam"
loss_DO = "mse"
dropout_prob_DO = 0.2  # probability of dropout for entire model

# (b) Gaussian Process Regression
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitGP = True  # compare to GP model?
#####################
kernel = "rbf"
whitekernel = False
constant_value = 4  # multiplicative constant factor
constant_value_bounds = (
    2,
    4,
)  #  bounds for hyperparameter optimization of multiplicative constant factor
length_scale = 0.01  # length scale of rbf kernel
length_scale_bounds = (
    length_scale,
    1e5,
)  # bounds for hyperparameter optimization of length scale of rbf kernel
noise_level = (
    1e-7 if whitekernel else None
)  # noise level for PIs (only in combination with whitekernel_GP)
noise_level_bounds = (
    (1e-10, 1e-5) if whitekernel else None
)  # bounds for hyperparameter optimization of noise level for PIs (only in combination with whitekernel_GP)
alpha = 0.001 if not whitekernel else 0  # noise level for CIs
n_restarts_optimizer = 10  # number of restarts for hyperparameter optimization
std_min = 1e-3  # minimal predictive standard deviation


# (c) Deep Ensembles (inherits some parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitDE = True  # Compare to Deep Ensemble
#####################
layers_DE = (din, 256, 1024, 512, 1)
epochs_DE = epochs
batch_size_DE = batch_size
l2reg_DE = l2reg  # L2-reg
seed_init_DE = seed_init
optimizer_DE = "Adam"
loss_DE = "mse"  # 'nll' or 'mse' currently (mse generates ensemble with only one mean output and does not learn data noise)
number_of_networks_DE = 5
softplus_min_var_DE = (
    1e-6  # minimum variance for numerical stability (only used if loss == nll)
)


# (d) Hyper Deep Ensembles (inherits some parameters from NOMU)
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
dropout_prob_HDE = (1e-3, 0.9)  # log uniform bounds
seed_init_HDE = 1
optimizer_HDE = "Adam"
clipnorm_HDE = None  # set maximal gradient value or set to None
loss_HDE = "mse"  # 'nll' or 'mse' currently (mse generates ensemble with only one mean output and does not learn data noise)
K_HDE = 5
kappa_HDE = 50
stratify_HDE = True  # paper=True
fixed_row_init_HDE = True  # paper=True
refit_HDE = False
softplus_min_var_HDE = (
    1e-6  # minimum variance for numerical stability (only used if loss == nll)
)
learning_rate_HDE = None


# (iv) SAVING & LOADING
#############################################################################################################################
savemodel = True  # save models and parameters?
loadmodel = False  # load models and parameters?


# %% (2) Data parameters
static_parameters = OrderedDict(
    zip(["function_name", "n_train", "n_aug"], [function_name, n_train, n_aug])
)
random_locations = True
# seed = 505  #(configuration for Figure 5 and Figure 16, respectively)

static_parameters["random_locations"] = random_locations
static_parameters["seed"] = SEED

(
    x,
    y,
    x_train,
    y_train,
    x_aug,
    y_aug,
    x_val,
    y_val,
    n_train,
    n_val,
) = generate_augmented_data(
    data="irradiance",
    df=df,
    train_size=train_size,
    seed=SEED,
    plot=True,
    n_aug=n_aug,
    random_aug=False,
    start=start,
    stop=stop,
    n_strips=n_strips,
    lenstrips=lenstrips,
)

# reset batch size in case we decreased number of training data
if train_size < 1:
    batch_size = n_train
    batch_size_BNN = batch_size
    batch_size_DE = batch_size
    batch_size_HDE = batch_size
    batch_size_DO = batch_size

# determine accurate l2_reg for DO and DE
l2reg_DO = l2reg_DO / n_train * (1 - dropout_prob_DO)
l2reg_DE = l2reg_DE / n_train

# %% (3) create save folder and/or enter load path
savepath = None
loadpath = None
if savemodel:
    foldername = "_".join([function_name, datetime.now().strftime("%d_%m_%Y_%H-%M-%S")])
    savepath = os.path.join(os.getcwd(), foldername)
    os.mkdir(savepath)  # if folder exists automatically an FileExistsError is thrown
if loadmodel:
    folder = input("Enter folder for loadpath: ")
    loadpath = os.path.join(os.getcwd(), folder)

# %% (4a) NOMU
start0 = datetime.now()

if FitNOMU:
    nomu = NOMU()
    if loadmodel:
        nomu.load_models(
            absolutepath=loadpath,
            model_numbers=1,  # give a list of model numbers for loading multiple NOMU models
            verbose=0,
        )
    else:
        nomu.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
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
            optimizer_clipnorm=clipnorm,
        )
        nomu.initialize_models(verbose=0)
        nomu.compile_models(verbose=0)
        nomu.fit_models(x=x, y=y, verbose=0)
        nomu.plot_histories(
            yscale="log",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(SEED)
                + start0.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )
    if savemodel:
        nomu.save_models(absolutepath=savepath)

if FitNOMUDj:
    nomu_dj = NOMU_DJ()
    if loadmodel:
        nomu_dj.load_models(
            absolutepath=loadpath,
            model_numbers=1,  # give a list of model numbers for loading multiple NOMU DJ models
            verbose=0,
        )
    else:
        nomu_dj.set_parameters(
            layers=layers,
            epochs=epochs,
            batch_size=batch_size,
            l2reg=l2reg,
            optimizer_name=optimizer,
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
            optimizer_clipnorm=clipnorm,
        )
        nomu_dj.initialize_models(verbose=0)
        nomu_dj.compile_models(verbose=0)
        nomu_dj.fit_models(x=x, y=y, verbose=0)
        nomu_dj.plot_histories(
            yscale="log",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(SEED)
                + start0.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )
    if savemodel:
        nomu_dj.save_models(absolutepath=savepath)
# %% (4b) MC Dropout
if FitMcDropout:
    mc_dropout = McDropout()
    if loadmodel:
        mc_dropout.load_models(
            absolutepath=loadpath,
            model_numbers=1,  # give a list of model numbers for loading multiple DO models
            verbose=0,
        )
    else:
        mc_dropout.set_parameters(
            layers=layers_DO,
            epochs=epochs_DO,
            batch_size=batch_size_DO,
            l2reg=l2reg_DO,
            optimizer_name=optimizer_DO,
            seed_init=seed_init_DO,
            loss=loss_DO,
            dropout_prob=dropout_prob_DO,
        )
        mc_dropout.initialize_models(verbose=0)
        mc_dropout.compile_models(verbose=0)
        mc_dropout.fit_models(x=x_train[:, :-1], y=y_train, verbose=0)
        mc_dropout.plot_histories(
            yscale="linear",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(SEED)
                + start0.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )
    if savemodel:
        mc_dropout.save_models(absolutepath=savepath)

# %% (4c) Gaussian Process
if FitGP:
    gp = GaussianProcess()
    if loadmodel:
        gp.load_models(
            absolutepath=loadpath,
            model_numbers=1,  # give a list of model numbers for loading multiple GP models
            verbose=0,
        )
    else:
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
        )
        gp.initialize_models(verbose=0)
        gp.compile_models(verbose=0)
        gp.fit_models(x=x_train[:, :-1], y=y_train, verbose=0)
    if savemodel:
        gp.save_models(absolutepath=savepath)

# %% (4d) Deep Ensemble
if FitDE:
    deep_ensemble = DeepEnsemble()
    if loadmodel:
        deep_ensemble.load_models(
            absolutepath=loadpath,
            model_numbers=1,  # give a list of model numbers for loading multiple DE models
            verbose=0,
        )
    else:
        deep_ensemble.set_parameters(
            layers=layers_DE,
            epochs=epochs_DE,
            batch_size=batch_size_DE,
            l2reg=l2reg_DE,
            optimizer_name=optimizer_DE,
            seed_init=seed_init_DE,
            loss=loss_DE,
            number_of_networks=number_of_networks_DE,
            softplus_min_var=softplus_min_var_DE,
        )
        deep_ensemble.initialize_models(verbose=0)
        deep_ensemble.compile_models(verbose=0)
        deep_ensemble.fit_models(x=x_train[:, :-1], y=y_train, verbose=0)
        deep_ensemble.plot_histories(
            yscale="linear",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(SEED)
                + start0.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )
    if savemodel:
        deep_ensemble.save_models(absolutepath=savepath)

end0 = datetime.now()
print(
    "\nTraining Time Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end0 - start0)),
    "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
)

#%% (4e) Hyper Deep Ensemble
if FitHDE:
    hyper_deep_ensemble = HyperDeepEnsemble()
    if loadmodel:
        hyper_deep_ensemble.load_models(
            absolutepath=loadpath, model_numbers=[1], verbose=0
        )
    else:
        hyper_deep_ensemble.set_parameters(
            layers=layers_HDE,
            epochs=epochs_HDE,
            batch_size=batch_size_HDE,
            l2reg=l2reg_HDE,
            optimizer_name=optimizer_HDE,
            seed_init=seed_init_HDE,
            loss=loss_HDE,
            dropout_prob=dropout_prob_HDE,
            K=K_HDE,
            kappa=kappa_HDE,
            test_size=test_size_HDE,
            stratify=stratify_HDE,
            fixed_row_init=fixed_row_init_HDE,
            refit=refit_HDE,
            softplus_min_var=softplus_min_var_HDE,
            optimizer_learning_rate=learning_rate_HDE,
            optimizer_clipnorm=clipnorm_HDE,
        )
        hyper_deep_ensemble.hyper_deep_ens(
            x=x_train[:, :-1],
            y=y_train,
            score=gaussian_nll_score,
            score_single_model=mse_score,
            random_state=SEED,
            verbose=1,
        )
        hyper_deep_ensemble.plot_histories(
            yscale="linear",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(SEED)
                + start0.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )
    if savemodel:
        hyper_deep_ensemble.save_models(absolutepath=savepath)

# %% (5) Set plot parameters
c_NOMU = 2
c_NOMU_DJ = 2
c_DO = 5
c_GP = 1
c_DE = 5
c_HDE = 20

bounds_variant_NOMU = "standard"
bounds_variant_NOMU_DJ = "standard"
bounds_variant_DO = "standard"
bounds_variant_GP = "standard"
bounds_variant_DE = "standard"
bounds_variant_HDE = "standard"


# (5a) parameters for ROC-like curves

linethreshy_ROC = 1
# linear grid
c_max_ROC = None
resolution_ROC = None
# custom grid
grid_min = 1e-3
grid_max = 1e3
steps = int(1e5)
factor = (grid_max / grid_min) ** (1 / steps)
custom_c_grid_ROC = np.array(
    [0]
    + [grid_min * factor ** i for i in range(steps)]
    + [grid_max * 2 ** k for k in range(1, 20)]
)
cp_max_ROC = 1  # maximum coverage probability as stopping criteria
captured_flag = False  # if true, calculate mw captured in ROC plot. else, mw.


# %% (6) Plot
# ('# CHOOSE' indicates parameters that can/should be changed manually in this part of the code)
plot_predictions(
    x_train=x_train,
    y_train=y_train,
    x_aug=x_aug,
    y_aug=y_aug,
    x_val=x_val,
    y_val=y_val,
    f_true=None,
    filepath=savepath if savemodel else loadpath,
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
    radPlot=1.01,  # CHOOSE
    save=True,  # CHOOSE
    markersize=4,  # CHOOSE
    transparency=0.5,  # CHOOSE
    linethreshy_ROC=linethreshy_ROC,
    c_max_ROC=c_max_ROC,
    custom_c_grid_ROC=custom_c_grid_ROC,
    cp_max_ROC=cp_max_ROC,
    resolution_ROC=resolution_ROC,
    show_details_title=False,  # CHOOSE
)

#%%
plot_irradiance(
    x_train=x_train,
    y_train=y_train,
    x_aug=x_aug,
    y_aug=y_aug,
    x_val=x_val,
    y_val=y_val,
    filepath=savepath if savemodel else loadpath,
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
    save=True,  # CHOOSE
    markersize=4,  # CHOOSE
    transparency=0.5,  # CHOOSE
    show_details_title=False,  # CHOOSE
)
