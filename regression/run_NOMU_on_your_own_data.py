# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 10:48:09 2022

This file is an example of how to run NOMU on your own real data.

"""

# Libs
import os
from datetime import datetime
import json
import numpy as np

# ------------------------------------------------------------------------- #
# disable eager execution for tf.__version__ 2.3.0
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------- #

# Own Modules
from algorithms.model_classes.nomu import NOMU
from algorithms.util import timediff_d_h_m_s

# %% DATA PREPARATION

# provided example data (stems from a GaussianBNN)
#############
x = np.asarray(json.load(open("x_data.json")))
y = np.asarray(json.load(open("y_data.json")))
n_train = x.shape[0]
input_dim = x.shape[1]
#############

# 1. scale training data: X to [-1,1]^input_dim, Y to [-1,1]^1
normalize_data = True  # recommended to set to True for better learning

if normalize_data:
    x_maxs = np.max(x, axis=0)
    x_mins = np.min(x, axis=0)
    y_max = np.max(y)
    y_min = np.min(y)

    for i, x_min in enumerate(x_mins):
        x[:, i] = 2 * ((x[:, i] - x_min) / (x_maxs[i] - x_min)) - 1

    y = 2 * ((y - y_min) / (y_max - y_min)) - 1

    print(f"\nScaled X-Training Data of shape {x.shape}")
    print(x)
    print(f"\nScaled y-Training Data of shape {y.shape}")
    print(y)

# 2. generate NOMU input: add artificial (also called augmented) input data points for NOMU-loss term (c); in this example sampled uniformly at random
#############
n_art = 200  # number of artificial (augmented) input data points.
#############

aug_in_training_range = False  # sample artificial training data only in training data range? If False, they are sampled from the normalized range.
aug_range_epsilon = 0.05

# find range to sample augmented data from
if aug_in_training_range:
    x_min_art = np.min(x, axis=0)
    x_max_art = np.max(x, axis=0)
else:
    x_min_art = -1
    x_max_art = 1
margin = (x_max_art - x_min_art) * aug_range_epsilon
x_min_art -= margin
x_max_art += margin


x_art = np.random.uniform(
    low=x_min_art, high=x_max_art, size=(n_art, x.shape[1])
)  # if you activate MCaug=True, these values do not matter because they will be overwritten internally by NOMU
y_art = np.ones((n_art, 1))  # these values do not matter, only the dimension matters
x = np.concatenate(
    (x, np.zeros((n_train, 1))), axis=-1
)  # add 0-flag identifying a real training point
x_art = np.concatenate(
    (x_art, np.ones((x_art.shape[0], 1))), axis=-1
)  # add 1-flag identifying a artificial training point

x_nomu = np.concatenate((x, x_art))
y_nomu = np.concatenate((np.reshape(y, (n_train, 1)), y_art))

print(
    f"\nX NOMU Input Data of shape {x_nomu.shape} (real training points:{n_train}/artificial training points:{n_art})"
)
print(x_nomu)
print(
    f"\ny NOMU Input Data of shape {y_nomu.shape} (real training points:{n_train}/artificial training points:{n_art})"
)
print(y_nomu)
# %% NOMU HPs
layers = (input_dim, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # layers incl. input and output
epochs = 2 ** 10
batch_size = 32
l2reg = 1e-8  # L2-regularization on weights of \hat{f} network
l2reg_sig = l2reg  # L2-regularization on weights of \hat{r}_f network
seed_init = 1  # seed for weight initialization

# (b) optimizer
# ----------------------------------------------------------------------------------------------------------------------------
optimizer = "Adam"  # select optimizer stochastic gradient descent: 'SGD' or adaptive moment estimation: 'Adam'

# (c) loss parameters
# ----------------------------------------------------------------------------------------------------------------------------
MCaug = True  # Monte Carlo approximation of the integrals in the NOMU loss with uniform sampling
mu_sqr = 0.1  # weight of squared-loss (\pi_sqr from paper)
mu_exp = 0.01  # weight exponential-loss (\pi_exp from paper)
c_exp = 30  # constant in exponential-loss
side_layers = (input_dim, 2 ** 10, 2 ** 10, 2 ** 10, 1)  # r-architecture
r_transform = "custom_min_max"  # either 'id', 'relu_cut' or 'custom_min_max' (latter two use r_min and r_max).
r_min = 1e-3  # minimum model uncertainty for numerical stability
r_max = 2  # asymptotically maximum model uncertainty
# %% RUN NOMU
start0 = datetime.now()
foldername = "_".join(["NOMU", "real_data", start0.strftime("%d_%m_%Y_%H-%M-%S")])
savepath = os.path.join(os.getcwd(), foldername)
os.mkdir(savepath)  # if folder exists automatically an FileExistsError is thrown
verbose = 0
#
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
    n_aug=n_art,
    mu_sqr=mu_sqr,
    mu_exp=mu_exp,
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
nomu.fit_models(
    x=x_nomu,
    y=y_nomu,
    verbose=verbose,
)
nomu.plot_histories(
    yscale="log",
    save_only=True,
    absolutepath=os.path.join(
        savepath, "Plot_History_seed_" + start0.strftime("%d_%m_%Y_%H-%M-%S")
    ),
)

end0 = datetime.now()
print(
    "\nTotal Time Elapsed: {}d {}h:{}m:{}s".format(*timediff_d_h_m_s(end0 - start0)),
    "(" + datetime.now().strftime("%H:%M %d-%m-%Y") + ")",
)
# %% HOW TO USE NOMU OUTPUTS
new_x = np.array([[0, 0], [0.5, 0.5]])  # 2 new input points
predictions = nomu.calculate_mean_std(new_x)  # predict mean and model uncertainty

mean, sigma_f = predictions["NOMU_Neural_Network_1"]  # extract them

if normalize_data:
    print(f"\nScaled-[-1,1]-Predictions mean:{mean} | sigma_f:{sigma_f}")

    mean_orig, sigma_f_orig = (y_max - y_min) * (mean + 1) / 2 + y_min, (
        y_max - y_min
    ) * (
        sigma_f + 1
    ) / 2 + y_min  # rescale them to original scale
    print(f"\nPredictions mean:{mean_orig} | sigma_f:{sigma_f_orig}")
else:
    print(f"\nPredictions mean:{mean} | sigma_f:{sigma_f}")
