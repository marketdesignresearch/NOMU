# NOMU:Neural Optimization-based Model Uncertainty

This is a piece of software used for performing experiments on NOMU uncertainty bounds and the three popular benchmarks (i) Monte-Carlo Dropout [1], (ii) Deep Ensembles [2] and (iii) Gaussian Process. The experiments are described in detail in the paper _NOMU: Neural Optimization-based Model Uncertainty_.


## A. Requirements

* Python>=3.7

## B. Dependencies

Prepare your python environment (`conda`, `virtualenv`, etc.) and enter this after activating your environment. For the regression experiments use the requirments.txt from the folder regression.  For the Bayesian optimization experiments use the requirments.txt from the folder bayesian_optimization.

Using pip:
```bash
$ pip install -r requirements.txt

```

**_NOTE:_** On some operating systems issues with tensorflow 2.3.0 can occur caused by the GPU.
These issue can be resolved by not using the GPU and calculate everything using the CPU. 
For this copy the code below to the top of the respective script:

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

## C. Regression Experiments

### How to run 

First navigate to the folder _regression_.

#### C.1 To run a regression experiment on any of the provided test functions over multiple seeds of synthetic data
1.  Set the desired test function, seeds and parameters in the file simulation_synthetic_functions.py. 
2.  Run:
    ```bash
    $ python simulation_synthetic_functions.py
    ```
Additionally to the console printout a folder Multiple_Seeds\_\<function_name\>\_\<date\>\_\<time\> will be created in the regression folder, where the UBs-plots and ROC-plots can be found.

**_NOTE:_** Parameters are set in python simulation_synthetic_functions.py such that one run for the Levy1D function is performed and one reconstructs **Figure 4** from the paper (runtime ~5 mins)

**_NOTE:_** To enable a head-to-head comparison of **Table 1** and **Table 2**:

1. We use for the 500 runs in 1D and 2D regression the seeds 501--1000 (unparallelized runtime on single local machine for one test function ~5*500min=42h). Concretely, we then set in simulation_synthetic_functions.py the parameters
    - _number_of_instances = 500_
    - _my_start_seed = 501_
2.  We conduct these experiments on
    - **system:** Linux
    - **version:** SMP Debian 4.19.160-2 (2020-11-28)
    - **platform:** Linux-4.19.0-13-amd64-x86_64-with-debian-10.8
    - **machines:** Intel Xeon E5-2650 v4 2.20GHz processors with 48 logical cores and 128GB RAM and Intel E5 v2 2.80GHz processors with 40 logical cores and 128GB RAM
    - **python:** Python 3.7.3 [GCC 8.3.0] on linux
3.  These experiments are conducted with Tensorflow using the CPU only and no GPU


#### C.2 To run the real data experiment (solar irradiance data interpolation [3]) 
1. Set the desired parameters in the file simulation_synthetic_functions.py.
2. Run:
    ```bash
    $ python simulation_solar_irradiance.py
    ```
When finished a folder called Irradiance\_\<date\>\_\<time\> will be created in the regression folder, where the UBs-plots and ROC-plots can be found.

**_NOTE:_** To enable a head-to-head comparison of **Figure 5**:

1. We set for this experiment the seed equal to 655 (runtime ~2h). Concretely, we set in simulation_solar_irradiance.py the parameter
    - _SEED=655_
2. These experiments were conducted on
    - **system:** Linux
    - **version:** Fedora release 32 (Thirty Two)
    - **platform:** Linux-5.8.12-200.fc32.x86_64-x86_64-with-glibc2.2.5
    - **machines:** Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz processors with 4 cores and 15GB RAM
    - **python:** Python 3.8.7 [GCC 10.2.1 20201125 (Red Hat 10.2.1-9)] on linux
3.  These experiments are conducted with Tensorflow using the CPU only and no GPU



## D. Bayesian Optimization Experiments

### How to run

To run the bayesian optimization experiment create or adjust the configuration file
which defines all parameters used for the experiment run.
Then save the configuration file as a '.ini' file.
Once the configuration file is prepared run the script "simulation_BO.py" located in the bayesian_optimization folder 
 with the path to the configuration file as first argument. 
For example like this:

```bash
$ python simulation_BO.py ./example_config.ini
```
**_NOTE:_** Parameters are set in /example_config.ini such that one run for the Levy5D function is performed for a _MW scaling_ of 0.5 and 64 BO steps. This reconstructs one run of **Figure 17 d** in the Appendix ((unparallelized runtime on single local machine without GPU tensorflow ~12h))

When using the DIRECT optimizer for the acquisition function optimization depending on the operating system
some additional steps are required to be able to run the scipydirect implementation of the direct algorithm.

First install the fortran compilers to you conda environment using (not required of Linux systems):

```bash
$ conda install -c msys2 m2w64-gcc
```

If the code still does not work, copy the file "bayesian_optimization/libs/direct.cp37-win_amd64.pyd"(Windows) or "bayesian_optimization/libs/direct.cpython-37m-x86_64-linux-gnu.so"(Linux) into the
scipydirect folder in your environment lib. Usually the folder is located under
'Anaconda<VERSION>/envs/<ENVNAME>/Lib/site-packages/scipydirect' (Windows with Anaconda).
Make sure the requirements from the requirements.txt are already installed.

**_NOTE:_** Runnung a Bayesian optimization experiment for multiple steps even for one single seed and a single test function can take
multiple hours (GPU supported tensorflow is thus recommended).

### Configuration
The Bayesian optimization experiment can be configured using a config file.
With this config file the different algorithms and all subprocesses can be configured.
The config file is from filetype (.ini) and looks like the following example:

```bash
[General]
seeds = 1

[BO]
function = levy5D
output_path = 
steps = 64
n_train = 8
lower_bounds = -1.0, -1.0, -1.0, -1.0, -1.0
upper_bounds = 1.0, 1.0, 1.0, 1.0, 1.0


[Optimizer]
optimizer = adam
learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-07
amsgrad = false

[Acquisition]
function = upper_bound
factor = 1.0

    [[mean_width_scaling_mc]]
    order=1
    scale_mean_width=0.5
    lower_bound=-1., -1., -1., -1., -1.
    upper_bound=1., 1., 1., 1., 1.
    n_test_points=20000
    once_only=true

[Acquisition Optimizer]
optimizer = direct
lower_search_bounds = -1.0, -1.0, -1.0, -1.0, -1.0
upper_search_bounds = 1.0, 1.0, 1.0, 1.0, 1.0

	[[dynamic_c_exp]]
	x_range = 2.0
	n_start_points = 8
	range_fraction = 0.25 
	end_eps = 0.01
	start_eps =
	max_increase_iter = 15
	n_steps = 64

[GP]
alpha=1e-7
optimizer=fmin_l_bfgs_b
n_restarts_optimizer=10
normalize_y=true
copy_X_train=true
random_state=
std_min=1e-6
kernel_once=false

    [[Kernel]]
    kernel=rbf
    constant_value=1
    constant_value_bounds=1e-5,1e5
    length_scale=1
    length_scale_bounds=1e-5,1e5

[NOMU]
epochs = 1024
r_max = 2.0
r_min = 1e-6
mip = false
main_layers = 5, 1024, 1024, 1024, 1
side_layers = 5, 1024, 1024, 1024, 1
lowerbound_x_aug = -1.0, -1.0, -1.0, -1.0, -1.0
upperbound_x_aug = 1.0, 1.0, 1.0, 1.0, 1.0
n_aug = 500
c_aug = 500
mu_sqr = 1.0
mu_abs = 0.0
mu_exp = 0.01
c_exp = 30
c_2 =
seed = 3
l2reg = 1e-8
activation = relu
RSN = false

[DO]
epochs = 1024
n_samples = 10
layers = 5, 1024, 2048, 1024, 1
activation = relu
RSN = false
dropout = 0.2
seed = 3
loss = mse
l2reg = 1e-8
normalize_regularization=true

[DE]
epochs = 1024
n_ensembles = 5
random_seed = true
layers = 5, 256, 1024, 512, 1
activation = relu
RSN = false
l2reg = 1e-8
softplus_min_var = 1e-6
s = 0.05
seed = 3
loss =
no_noise = true
normalize_regularization=true
```

Each section defines a certain part of the whole algorithm (indicated in the config file with "[]").
In the following, we highlight for each section which parameters and submodules can be configured and give explanations of all parameters (relevant for the paper).


#### General (required)
This section defines the general setup of the experiment.
In particular, it defines the number of runs and the seeds for these runs.

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| seeds      | list of integers: defines the seeds for the different runs which set the random starting samples; the number of intergers in the list defines how many runs will be conducted | 1,2,3,4,5 | No |

#### BO (required)
This section defines the setup of the Bayesian optimization.
Which function to test, how many steps to take and how many initial samples to use.

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| function      | string defining which function to use (full list of possible strings below) | levy5D | No |
| output_path      | path where to store the output files | ./some_path | No |
| steps      | integer defining ho many steps the Bayesian optimization should take | 64 | No |
| n_train      | integer defining how many starting samples should be sampled |8 | No |
| lower_bounds      | list of integers which define the lower bounds of the input space | -1.0, -1.0, -1.0, -1.0, -1.0 | No |
| upper_bounds      | list of integers which define the upper bounds of the input space | 1.0, 1.0, 1.0, 1.0, 1.0 | No |

Possible functions to use:
- forrester
- levy
- sinone
- branin2D
- camelback2D
- goldstein_price2D
- levy5D
- levy10D
- levy20D
- rosenbrock2D
- rosenbrock5D
- rosenbrock10D
- rosenbrock20D
- perm2D
- perm5D
- perm10D
- perm20D
- g_function2D
- g_function5D
- g_function10D
- g_function20D
- schwefel3D
- hartmann3D
- hartmann6D
- michalewicz2D
- michalewicz5D
- michalewicz10D
- michalewicz20D

#### Acquisition (required)

This section defines the acquisition function that should be used for the Bayesian optimization

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| function      | string defining which acquisition function to use (full list off possible options below) | upper_bound | No |
| factor      | factor to multiply the uncertainty width for Upper Bounds | 1 | required for upper bound |
| xi      | "trade of" parameter for Probability of Improvement (PoI) and Expected Improvement (EI)  | 0.1 | required for PoI and EI |

Possible acquisition functions to use:
- mean_only (resolve to the mean prediction value only)
- uncertainty_only (resolve to the uncertainty prediction value only)
- upper_bound (upper bound (mean + uncertainty))
- probability_of_improvement (probability of improvement (PoI))
- expected_improvement (expected improvement (EI))

##### Extensions
For the acquisition functions there are different extensions that can be made. These extensions are wrapped around the acquisition function
and modify it. These extension can be configured using subsection indicated with [[]]

###### mean_width_scaling
Estimates the mean width (MW) of the predicted uncertainty bounds based on a grid.

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| scale_mean_width      | float defining the MW budget which defines the calibration parameter c that is used to scale the uncertainty. | 0.05 | No |
| lower_bound      | list of integers which defines the lower bounds of the input space for the test points | -1.0, -1.0, -1.0, -1.0, -1.0 | No |
| upper_bound      | list of integers which defines the upper bounds of the input space for the test points | 1.0, 1.0, 1.0, 1.0, 1.0 | No |
| n_test_points      | number of test points per input dimension (total number of grid points = n_test_points^d) | 200 | No |
| once_only      | boolean defining that the calibration parameter c should only be calculated in the first step | true | No |
| order      | order in which the extension should be applied for the case that there are multiple extensions active | 1 | No |

###### mean_width_scaling_mc
Estimates the mean width (MW) of the predicted uncertainty bounds based on Monte Carlo sampling.

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| scale_mean_width      | float defining the MW budget which defines the calibration parameter c that is used to scale the uncertainty. | 0.05 | No |
| lower_bound      | list of integers which defines the lower bounds of the input space for test points | -1.0, -1.0, -1.0, -1.0, -1.0 | No |
| upper_bound      | list of integers which defines the upper bounds of the input space for test points | 1.0, 1.0, 1.0, 1.0, 1.0 | No |
| n_test_points      | number of samples for the Monte Carlo sampling | 20000 | No |
| once_only      | boolean defining that the calibration parameter c should only be calculated in the first step | true | No |
| order      | order in which the extension should be applied for the case that there are multiple extensions active | 1 | No |

###### bounded_r
Applies a bounding function to the uncertainty. The NOMU method uses this bounding by default and thus it does not need to be configured here.

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| r_max      | upper bound for the uncertainty | 2.0 | No |
| r_min      | lower bound for the uncertainty | 1e-6 | No |
| order      | order in which the extension should be applied for the case that there are multiple extensions active | 1 | No |

#### Acquisition Optimizer (required)

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| optimizer      | string defining which acquisition function optimizer should be used (full list below) | direct | No |
| lower_search_bounds      | lower bound for the uncertainty | 0.01 | No |
| upper_search_bounds      | upper bound for the uncertainty | 2 | No |
| order | order in which the extension should be applied for the case that there are multiple extensions active | 1 | No |

Possible acquisition functions optimizer:
- grid_search 
- direct (DIRECT)
- mip (Mixed Integer Programming (only for NOMU))

##### Extensions
For the acquisition function optimizer there are different extensions that can be applied. These extensions are wrapped around the acquisition function optimizer and modify it. These extension can be configured using subsection indicated with [[]] in the config file.

###### dynamic_c_exp
Specifies the parameters of the dynamic c procedure with an exponential decay (see Appendix B.3.2.; Note the delta from the appendix is the epsilon here). 

Note: if start_eps is not specified it is calculated as follows:
start_eps = (x_range * range_fraction) / n_start_points

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| x_range | input range | 2 | No |
| n_start_points | number of starting samples of the run | 8 | No |
| range_fraction | fraction of the input space | 0.25 | No |
| end_eps | the last value for the epsilon | 0.01 | No |
| start_eps | the starting value for the epsilon |  | Yes |
| max_increase_iter | number of times the dynamic c can double the factor c | 15 | No |
| n_steps | number of steps the epsilon decays (e.g.: for 64 total steps and "n_step"=60, the last 4 steps will be with smallest epsilon) | 60 | No |


###### dynamic_c
Specifies the parameters of the dynamic c procedure with a linear decay.

Note: if start_eps is not specified it is calculated as follows:
start_eps = (x_range * range_fraction) / n_start_points

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| x_range | input range | 2 | No |
| n_start_points | number of starting samples of the run | 8 | No |
| range_fraction | fraction of the input space | 0.25 | No |
| end_eps | the last value for the epsilon | 0.01 | No |
| start_eps | the starting value for the epsilon |  | Yes |
| max_increase_iter | number of times the dynamic c procedure can double the factor c | 15 | No |
| n_steps | number of steps the epsilon decays (e.g.: for 64 total steps and "n_step"=60, the last 4 steps will be with smallest epsilon) | 60 | No |

#### Optimizer (required)

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| optimizer | string defining which NN optimizer to use (currently only adam supported) | adam | No |
| learning_rate | learning rate of the optimizer | 0.001 | No |
| beta_1 | beta_1 of the optimizer | 0.9 | No |
| beta_2 | beta_2 of the optimizer | 0.999 | No |
| epsilon | epsilon of the optimizer | 1e-07 | No |
| amsgrad | amsgrad of the optimizer | false | No |

#### NOMU
This section defines the parameters for NOMU algorithm

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| epochs | number of epochs to train the NOMU architecture | 1024 | No |
| r_max | upper bound for the readout map (\ell_max) | 2.0 | No |
| r_min | lower bound for the readout map (\ell_min) | 1e-6 | No |
| mip | boolean whether to use the mip compatible readout map | false | No |
| main_layers | numbers of nodes per layer for the main-network, as list | 5, 1024, 1024, 1024, 1 | No |
| side_layers | numbers of nodes per layer for the side-network, as list | 5, 1024, 1024, 1024, 1 | No |
| lowerbound_x_aug | lower bounds of input space for the artificial input points | -1.0, -1.0, -1.0, -1.0, -1.0 | No |
| upperbound_x_aug | upper bounds of input space for the artificial input points | 1.0, 1.0, 1.0, 1.0, 1.0 | No |
| n_aug | number of artificial input points | 500 | No |
| mu_sqr | factor for the squared term of the loss (pi_sqr) | 1.0 | No |
| mu_exp | factor for the exponential term of the loss (pi_exp) | 0.01 | No |
| c_exp | c for the exponential term of the loss | 30 | No |
| seed | seed for layer initialization | 3 | No |
| l2reg | L2-regularization parameter | 1e-8 | No |
| activation | string indicating which activation functions to use | relu | No |

#### DE
This section defines the parameters for the deep ensembles algorithm

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| epochs | number of epochs to train the networks | 1024 | No |
| n_ensembles | size of the ensemble | 5 | No |
| layers | numbers of nodes per layer for the network, as list | 5, 256, 1024, 512, 1 | No |
| normalize_regularization | should the L2-regularization be adjusted according to the data noise assumption | true | No |
| no_noise | should the method be adjusted for the noiseless case: If true loss=mse and each network has only one output (mean prediction), if false loss=nll and each network has two outputs (mean and data noise prediction) | true | No |
| loss | overwrite default losses defined be no_noise parameter with custom loss |   | Yes |
| softplus_min_var | softplus minimum variance (used for nll if no_noise is false) | 1e-6 | Yes |
| seed | seed for layer initialization | 3 | No |
| l2reg | L2-regularization parameter | 1e-8 | No |
| activation | string indicating which activation functions to use | relu | No |

#### DO
This section defines the parameters for the Monte Carlo (MC) dropout algorithm

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| epochs | number of epochs to train the network | 1024 | No |
| n_samples | number of stochastic forward passes | 10 | No |
| layers | numbers of nodes per layer for the network, as list | 5, 256, 1024, 512, 1 | No |
| normalize_regularization | should regularization be adjusted according to the data noise assumption | true | No |
| dropout | dropout probability | 0.2 | No |
| loss | overwrite with custom loss | mse | No |
| seed | seed for layer initialization | 3 | No |
| l2reg | L2-regularization factor | 1e-8 | No |
| activation | string indicating which activation functions to use | relu | No |

#### GP
This section defines the parameters for the Gaussian Process (GP) algorithm

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| alpha | inhereted parameter of the GaussianProcessRegressor from sklearn | 1e-7 | No |
| optimizer | inhereted parameter of the GaussianProcessRegressor from sklearn | fmin_l_bfgs_b | No |
| n_restarts_optimizer | inhereted parameter of the GaussianProcessRegressor from sklearn | 10 | No |
| normalize_y | inhereted parameter of the GaussianProcessRegressor from sklearn | true | No |
| copy_X_train | inhereted parameter of the GaussianProcessRegressor from sklearn | true | No |
| random_state | inhereted parameter of the GaussianProcessRegressor from sklearn |  | Yes |
| std_min | inhereted parameter of the GaussianProcessRegressor from sklearn | 1e-6 | No |
| kernel_once | can the kernel optimizer optimize the parameters only during the first BO step | false | No |

##### Kernel

The kernel for the Gaussian Process can be configured individually

| parameter        | Explanation | Example  |Can be empty  |
| ------------- |:-------------:| -----:|-----:|
| kernel | string defining which kernel to use (currently only option is rbf) | rbf | No |
| constant_value | starting value for the constant value | 1 | No |
| constant_value_bounds | bounds inside which the kernel optimizer can optimize the constant value parameter | 1e-5,1e5 | No |
| length_scale | starting value for the length scale | 1 | No |
| length_scale_bounds | bounds inside which the kernel optimizer can optimize the length scale parameter | 1e-5,1e5 | No |


**_NOTE:_** To enable a head-to-head comparison of **Table 3** and **Figure 6**:

1. We use seeds 1,...,100 for the 5D experiments and 1,...,50 for the 10D and 20D experiments (unparallized runtime on a single local machine for _one_ seed and _one_ test functions 12h~(5D) 18h~(10D) 24h~(20D)).
2.  We conduct these experiments on
    - **system:** Linux
    - **version:** SMP Debian 4.19.160-2 (2020-11-28)
    - **platform:** Linux-4.19.0-13-amd64-x86_64-with-debian-10.8
    - **machines:** Intel Xeon E5-2650 v4 2.20GHz processors with 48 logical cores and 128GB RAM and Intel E5 v2 2.80GHz processors with 40 logical cores and 128GB RAM
    - **python:** Python 3.7.3 [GCC 8.3.0] on linux
3.  These experiments are conducted with Tensorflow using the CPU only and no GPU


## E. References 

[1] Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning http://proceedings.mlr.press/v48/gal16.html

[2] Simple and Scalable Predictive UncertaintyEstimation using Deep Ensembles http://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf

[3] Total solar irradiance during the Holocene https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2009GL040142
