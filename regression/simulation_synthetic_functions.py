# -*- coding: utf-8 -*-
"""

This file is used for the regression experiments on synthetic test functions.

"""

# Libs
import os
import random
from collections import OrderedDict
from datetime import datetime
import pickle
import numpy as np
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
from plot_functions.plot_functions import plot_predictions, plot_predictions_2d
from data_generation.data_generator import generate_augmented_data
from data_generation.function_library import function_library
from algorithms.util import timediff_d_h_m_s
from performance_measures.load_and_print_results import load_and_print_results

__author__ = 'Hanna Wutte, Jakob Weissteiner, Jakob Heiss'
__copyright__ = 'Copyright 2020, NOMU: Neural Optimization-based Model Uncertainty'
__license__ = 'AGPL-3.0'
__version__ = '0.1.0'
__maintainer__ = 'Hanna Wutte, Jakob Weissteiner, Jakob Heiss'
__email__ = 'hanna.wutte@math.ethz.ch, weissteiner@ifi.uzh.ch, jakob.heiss@math.ethz.ch'
__status__ = 'Dev'
# %% (1) Model parameters

# (i) GROUND TRUTH FUNCTION
#############################################################################################################################

din = 1  # input dimension; specify ground truth function below

function_name = "Levy"
f_true = function_library(function_name)


# (ii) DATA
#############################################################################################################################
n_train = 2 ** 3  # number of training points.
n_aug = 2 ** 7  # number of augmented points.

# For equidistant sampling: n_train, n_aug should be given in powers of din
# (i.e., 2**(x*din) datapoints --> 2**x resolution in each dimension)


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
l2reg = 1e-8  # L2-regularization on weights of \hat{f} network
l2reg_sig = l2reg # L2-regularization on weights of \hat{r}_f network
seed_init = 30  # Seed for weight initialization (configuration for Figure 4; set to None for Table 1 and Table 2, respectively)

# (a2) NOMU DISJOINT (inherits all parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
# set to True for training NOMU without backward passes in dashed connections from \hat{f} network to \hat{r}_f network
#####################
FitNOMUDj = False
#####################

# (b) optimizer
# ----------------------------------------------------------------------------------------------------------------------------
optimizer = "Adam"  # select optimizer stochastic gradient descent: 'SGD' or adaptive moment estimation: 'Adam'

# (c) loss parameters
# ----------------------------------------------------------------------------------------------------------------------------
MCaug = False  # Monte Carlo Approximation of the Integrals in the specified Loss with uniform sampling?
mu_sqr = 0.1  # weight L2-loss for training data points (pi_sqr from paper)
mu_exp = 0.01  # weight exp-loss (pi_exp from paper)
c_exp = 30  # constant in exp loss
side_layers = (din, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # r-architecture
r_transform = "custom_min_max"  # either 'id', 'relu_cut' or 'custom_min_max' (latter 2 use r_min and r_max).
r_min = 1e-3  # minimum r for numerical stability
r_max = 2  # asymptotically maximum r

# (iv) BENCHMARKS:
#############################################################################################################################

# (a) MC Dropout (inherits some parameters from NOMU)
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
constant_value = 4  # multiplicative constant factor
constant_value_bounds = (
    4,
    4,
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


# (c) Deep Ensembles (inherits some parameters from NOMU)
# ----------------------------------------------------------------------------------------------------------------------------
#####################
FitDE = True  # Compare to Deep Ensemble
#####################
layers_DE = (din, 256, 1024, 512, 1)
epochs_DE = epochs
batch_size_DE = batch_size
l2reg_DE = l2reg / n_train  # L2-reg for ensembles
seed_init_DE = seed_init
optimizer_DE = "Adam"
loss_DE = "mse"  # 'nll' or 'mse' currently (mse generates ensemble with only one mean output and does not learn data noise)
number_of_networks_DE = 5
softplus_min_var_DE = (
    1e-6  # minimum variance for numerical stability (only used if loss == nll)
)

# (v) SAVING
#############################################################################################################################

foldername = "_".join(
    ["Multiple_Seeds", function_name, datetime.now().strftime("%d_%m_%Y_%H-%M-%S")]
)
savepath = os.path.join(os.getcwd(), foldername)
os.mkdir(savepath)  # if folder exists automatically an FileExistsError is thrown


# %% (2) Data parameters
####################################
number_of_instances = 1  # (configuration for Figure 4; set to 500 for Table 1 and Table 2, respectively)
my_start_seed = 513  # (configuration for Figure 4; set to 501 for Table 1 and Table 2, respectively)
seeds = [i for i in range(my_start_seed, number_of_instances + my_start_seed)]
####################################

static_parameters = OrderedDict(
    zip(
        ["function_name", "dim", "n_train", "n_aug"],
        [function_name, din, n_train, n_aug],
    )
)
noise_scale = 0  # noise scale for trainind data
random_locations = True  # random input training points?

static_parameters["noise_scale"] = noise_scale
static_parameters["random_locations"] = random_locations


# %% (3) Bounds parameters
c_NOMU = 1
c_NOMU_DJ = 1
c_DO = 4
c_GP = 1
c_DE = 15

bounds_variant_NOMU = "standard"
bounds_variant_NOMU_DJ = "standard"
bounds_variant_DO = "standard"
bounds_variant_GP = "standard"
bounds_variant_DE = "standard"


# (3a) parameters for ROC-like curves

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


# %% (4) Run Simulation
start0 = datetime.now()
#
verbose = 0
results = {}
instance = 1


for seed in seeds:

    # REPRODUCABILITY
    #------------------------------------------------------------------------------------------------------
    tf.compat.v1.keras.backend.clear_session()

    # 1. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 2. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 3. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)

    # 4. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    #------------------------------------------------------------------------------------------------------

    experiment_summary_csv = OrderedDict()
    start1 = datetime.now()
    static_parameters["seed"] = seed
    print(
        "\nInstance {}/{}: generate data for seed {}".format(instance, len(seeds), seed)
    )
    print("**************************************************************************")
    x, y, x_train, y_train, x_aug, y_aug, x_val, y_val = generate_augmented_data(
        din=din,
        n_train=n_train,
        n_val=100,
        f_true=f_true,
        random=random_locations,
        noise_scale=noise_scale,
        seed=seed,
        plot=False,
        batch_size_sampling=2 ** 16,
        n_aug=n_aug,
        random_aug=MCaug,
        noise_on_validation=0,
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
        )
        nomu.initialize_models(verbose=verbose)
        nomu.compile_models(verbose=verbose)
        nomu.fit_models(x=x, y=y, verbose=verbose)
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
        )
        nomu_dj.initialize_models(verbose=verbose)
        nomu_dj.compile_models(verbose=verbose)
        nomu_dj.fit_models(x=x, y=y, verbose=verbose)
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
            seed_init=seed_init_DO,
            loss=loss_DO,
            dropout_prob=dropout_prob_DO,
        )
        mc_dropout.initialize_models(verbose=verbose)
        mc_dropout.compile_models(verbose=verbose)
        mc_dropout.fit_models(x=x_train[:, :-1], y=y_train, verbose=verbose)
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
            seed_init=seed_init_DE,
            loss=loss_DE,
            number_of_networks=number_of_networks_DE,
            softplus_min_var=softplus_min_var_DE,
        )
        deep_ensemble.initialize_models(verbose=verbose)
        deep_ensemble.compile_models(verbose=verbose)
        deep_ensemble.fit_models(x=x_train[:, :-1], y=y_train, verbose=verbose)
        deep_ensemble.plot_histories(
            yscale="linear",
            save_only=True,
            absolutepath=os.path.join(
                savepath,
                "Plot_History_seed{}_".format(seed)
                + start1.strftime("%d_%m_%Y_%H-%M-%S"),
            ),
        )

    print("\nCalculate ROC & create plots")
    print("**************************************************************************")

    if din == 1:
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
            #
            radPlot=1.1,  # CHOOSE
            save=True,  # CHOOSE
            markersize=8,  # CHOOSE
            transparency=0.5,  # CHOOSE
            linewidth=1,  # CHOOSE
            # plotaugmented=False, # CHOOSE
            logy_ROC=True,  # CHOOSE
            linethreshy_ROC=linethreshy_ROC,
            c_max_ROC=c_max_ROC,
            custom_c_grid_ROC=custom_c_grid_ROC,
            cp_max_ROC=cp_max_ROC,
            resolution_ROC=resolution_ROC,
            show_details_title=False,  # CHOOSE
        )

    elif din == 2:
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
            # 1d also
            radPlot=1.1,  # CHOOSE
            save=True,  # CHOOSE
            resolution=50,  # CHOOSE
            markersize=60,  # CHOOSE
            linewidth=1,  # CHOOSE
            linethreshy_ROC=linethreshy_ROC,
            c_max_ROC=c_max_ROC,
            custom_c_grid_ROC=custom_c_grid_ROC,
            cp_max_ROC=cp_max_ROC,
            resolution_ROC=resolution_ROC,
            show_details_title=False,  # CHOOSE
            # 2d new
            colorlimits=[0, 2],  # CHOOSE
            only_uncertainty=True,  # CHOOSE
        )

    end1 = datetime.now()
    print(
        "\nInstance Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end1 - start1)),
        "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
    )
    instance += 1

# save info
with open(os.path.join(savepath, "results.pkl"), "wb") as f:
    pickle.dump(results, f)
f.close()


end0 = datetime.now()
print(
    "\nTotal Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end0 - start0)),
    "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
)


# %% SUMMARY TO CONSOLE
load_and_print_results(savepath, seeds)
