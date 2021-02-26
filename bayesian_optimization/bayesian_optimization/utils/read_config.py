from bayesian_optimization.functions import *
from bayesian_optimization.context.context import Context
from copy import deepcopy
from bayesian_optimization.nn_models.nomu_model import NOMUModel
from bayesian_optimization.estimators.single import SingleMethod
from bayesian_optimization.estimators.single_bounded import SingleMethodBounded
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction
from bayesian_optimization.functions import *
from bayesian_optimization.context.context import Context
from bayesian_optimization.context.inspector import Inspector
from bayesian_optimization.acq_optimizer.acq_optimizer import ACQOptimizer
from bayesian_optimization.nn_models.dropout_model import DropoutModel
from bayesian_optimization.estimators.sample import SampleMethod
from bayesian_optimization.nn_models.deep_ensemble_model import DeepEnsembleModel
from bayesian_optimization.estimators.gp import GP
from bayesian_optimization.estimators.ensemble import EnsembleMethod
from bayesian_optimization.data_gen.generate_samples import get_new_samples
from configobj import ConfigObj
from tensorflow.keras.optimizers import Adam
import numpy as np

def setup_method_specifics(method, config, context):
    if "Optimizer" in config:
        context.set_model_optimizer(read_optimizer(config))
    if "Acquisition Optimizer" in config:
        print("Acquisition Optimizer", config["Acquisition Optimizer"])
        optimizer_ub = method.SUPPORTED_ACQ_OPTIMIZER[config["Acquisition Optimizer"]["optimizer"]].read_from_config(
            config["Acquisition Optimizer"])
        context.set_model_optimizer(optimizer_ub)
    if "Acquisition" in config:
        context.set_acq(AcquisitionFunction.read_acq(config["Acquisition"], context))


def read_optimizer(config):
    supported_optimizers = {
        "adam": Adam,
    }
    assert "Optimizer" in config, "config file must include 'Optimizer' section"
    optimizer = supported_optimizers[config["Optimizer"]["optimizer"]](
        learning_rate=config["Optimizer"].as_float("learning_rate"),
        beta_1=config["Optimizer"].as_float("beta_1"),
        beta_2=config["Optimizer"].as_float("beta_2"),
        epsilon=config["Optimizer"].as_float("epsilon"),
        amsgrad=config["Optimizer"].as_bool("amsgrad")
    )
    return optimizer


def read_BO(config, seed):
    supported_function = {
        "forrester": Forrester,
        "levy": Levy,
        "sinone": SinOne,
        "branin2D": Branin2D,
        "camelback2D": Camelback2D,
        "goldstein_price2D": GoldsteinPrice,
        "levy5D": Levy5D,
        "levy10D": Levy10D,
        "levy20D": Levy20D,
        "rosenbrock2D": Rosenbrock2D,
        "rosenbrock5D": Rosenbrock5D,
        "rosenbrock10D": Rosenbrock10D,
        "rosenbrock20D": Rosenbrock20D,
        "perm2D": Perm2D,
        "perm5D": Perm5D,
        "perm10D": Perm10D,
        "perm20D": Perm20D,
        "g_function2D": GFunction2D,
        "g_function5D": GFunction5D,
        "g_function10D": GFunction10D,
        "g_function20D": GFunction20D,
        "schwefel3D": Schwefel3D,
        "hartmann3D": Hartmann3D,
        "hartmann6D": Hartmann6D,
        "michalewicz2D": Michalewicz2D,
        "michalewicz5D": Michalewicz5D,
        "michalewicz10D": Michalewicz10D,
        "michalewicz20D": Michalewicz20D,

    }
    print("config", config)
    assert "BO" in config, "config file must include 'BO' section"
    function = supported_function[config["BO"]["function"]]
    context = Context(callback=function.evaluate_scaled)
    samples_x = get_new_samples(seed, config)

    return context, function, samples_x


def setup_NOMU(config, context, inspector):
    context_NOMU = deepcopy(context)
    ub_model = NOMUModel.read_from_config(config["NOMU"])
    context_NOMU.set_estimator(SingleMethodBounded(
        config["NOMU"].as_int("epochs"),
        config["NOMU"].as_float("r_max"),
        config["NOMU"].as_float("r_min"),
        config["NOMU"].as_bool("mip"),
    ))
    context_NOMU.set_network_model(ub_model)
    context_NOMU.set_inspector(deepcopy(inspector))
    setup_method_specifics(NOMUModel, config["NOMU"], context_NOMU)
    return context_NOMU


def setup_DO(config, context, inspector):
    context_DO = deepcopy(context)
    do_model = DropoutModel.read_from_config(config["DO"])
    context_DO.set_estimator(SampleMethod(
        epochs=config["DO"].as_int("epochs"),
        base_l2_reg=config["DO"].as_float("l2reg"),
        n_samples=config["DO"].as_int("n_samples"),
        normalize_regularization=config["DO"].as_bool("normalize_regularization"),
    ))
    context_DO.set_network_model(do_model)
    context_DO.set_inspector(deepcopy(inspector))
    setup_method_specifics(DropoutModel, config["DO"], context_DO)
    return context_DO


def setup_DE(config, context, inspector):
    de_model = DeepEnsembleModel.read_from_config(config["DE"])
    context_DE = deepcopy(context)
    context_DE.set_estimator(EnsembleMethod(
        n_ensembles=config["DE"].as_int("n_ensembles"),
        epochs=config["DE"].as_int("epochs"),
        random_seed=config["DE"].as_bool("random_seed"),
        normalize_regularization=config["DE"].as_bool("normalize_regularization")
    ))
    context_DE.set_network_model(de_model)
    context_DE.set_inspector(deepcopy(inspector))

    if "optimizer" in config["DE"]:
        optimizer_de = DeepEnsembleModel.SUPPORTED_ACQ_OPTIMIZER[config["DE"]["optimizer"]].read_from_config(config)
        context_DE.set_model_optimizer(optimizer_de)
    return context_DE


def setup_GP(config, context, inspector):
    context_GP = deepcopy(context)
    context_GP.set_estimator(GP.read_from_config(config["GP"]))
    context_GP.set_inspector(deepcopy(inspector))
    return context_GP