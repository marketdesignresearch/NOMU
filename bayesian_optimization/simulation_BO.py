"""
Script for running the Bayesian Optimization according to the configuration defined in a config file (.ini).
Run the whole process and saves the results into the folder specified in the config file.
"""

# Libs
import sys
import time
import os
import dill as pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.keras import backend as K
# ------------------------------------------------------------------------- #
# disable eager execution for tf.__version__ 2.4.1
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
# ------------------------------------------------------------------------- #


# nomubo
from bayesian_optimization.bo.bo import BO
from bayesian_optimization.utils.read_config import *
from bayesian_optimization.utils.analytics import *
from bayesian_optimization.utils.utils import *
from bayesian_optimization.context.inspector import Inspector
from bayesian_optimization.acquisition.acquisition_function import AcquisitionFunction
from bayesian_optimization.acq_optimizer.acq_optimizer import ACQOptimizer

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#%%

method_names = {
    "SampleMethod": "MC Dropout",
    "SingleMethodBounded": "NOMU",
    "SingleMethodBoundedDropout": "NOMUD",
    "GP": "Gaussian Process",
    "EnsembleMethod": "Deep Ensemble",
    "HyperDeepEnsembleMethod": "Hyper Deep Ensemble"
}

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def run(cont, steps, function, output_path, seed):

    # REPRODUCABILITY
    #------------------------------------------------------------------------------------------------------
    tf.compat.v1.keras.backend.clear_session()
    # Apparently you may use different seed values at each stage
    SEED = seed

    # 1. Set `python` built-in pseudo-random generator at a fixed value
    random.seed(SEED)

    # 2. Set `numpy` pseudo-random generator at a fixed value
    np.random.seed(SEED)

    # 3. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(SEED)

    # 4. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    #------------------------------------------------------------------------------------------------------

    bo = BO(context=cont, maxiter=steps)
    max_y, arg_max_x = bo.run(function.evaluate_scaled, False, seed)
    if cont.nn_model is not None:
        cont.nn_model.model = None
        cont.nn_model.loss = None
    cont.estimator.models = None
    cont.estimator.weighted_models = None
    cont.estimator.HPE = None
    if output_path is not None:
        #tf.compat.v1.enable_eager_execution()
        os.makedirs(output_path, exist_ok=True)
        cont.estimator.context = None
        cont.inspector.context = None
        cont.acq_optimizer.context = None
        cont.model_optimizer = None
        pickle.dump(cont, open("{}/context.pickle".format(output_path), "wb"))
    K.clear_session()
    all_regrets = []
    for i in range(0,steps+1):
        all_regrets.append(np.min([function.get_regret(x) for x in cont.samples_y[:len(cont.samples_y)-steps+i]]))
    return function.get_regret(max_y), all_regrets


def loop_methods(ctxt, conf, insp):
    all_context = []
    names = []
    mean_width_scales = []

    if "GP" in config:
        all_context.append(setup_GP(conf, ctxt, insp))
        names.append("GP")

    if "DO" in config:
        all_context.append(setup_DO(conf, ctxt, insp))
        names.append("DO")

    if "DE" in config:
        all_context.append(setup_DE(conf, ctxt, insp))
        names.append("DE")

    if "HDE" in config:
        all_context.append(setup_HDE(conf, ctxt, insp))
        names.append("HDE")

    if "NOMU" in config:
        all_context.append(setup_NOMU(conf, ctxt, insp))
        names.append("NOMU")

    if "NOMUD" in config:
        all_context.append(setup_NOMUD(conf, ctxt, insp))
        names.append("NOMUD")

    return all_context, names

if __name__ == "__main__":
    print("Starting simulation...")
    ini_path = sys.argv[1]
    config = ConfigObj(ini_path)
    print("-....", config["General"].as_list("seeds"))
    seeds = np.array([int(i) for i in config["General"].as_list("seeds")])
    all_seed_regrets = {}
    all_seed_regrets_per_step = {
        "method": [],
        "step": [],
        "seed": [],
        "regret": [],
    }
    inspector = {
        'lower_bound': config["BO"]["lower_bounds"],
        'upper_bound': config["BO"]["upper_bounds"],
        'n_test_points': 1,
        'do_estimate_test_data': 'false',
        'do_inspect_estimation': 'false',
        'do_inspect_optimization': 'false',
        'do_inspect_acq': 'false',
        'store_estimators': 'false',
        'inspector_path': config["BO"]["output_path"],
    }
    config_inspector = ConfigObj()

    config_inspector["Inspector"] = inspector
    for seed in seeds:
        print("starting simulation with seed {}...".format(seed))
        context, function, samples_x = read_BO(config, seed)
        samples_y = function.evaluate_scaled(samples_x)
        context.set_samples(samples_x, samples_y)
        context.set_acq(AcquisitionFunction.read_acq(config["Acquisition"], context))
        context.set_acq_optimizer(ACQOptimizer.read_acq_optimizer(config["Acquisition Optimizer"]))
        context.set_model_optimizer(read_optimizer(config))
        inspector = Inspector.read_inspector(config_inspector)

        mean_width_scales = []
        if "parallel run" in config:
            print("Running in parallel Mode...")
            if "mean_width_scales" in config["parallel run"]:
                mean_width_scales = [float(i) for i in config["parallel run"].as_list("mean_width_scales")]
                print("Running mean width scales in parallel: ", mean_width_scales)

        base_path = config_string_or_none(config["BO"]["output_path"])
        #print("Output will be stored here: ", base_path)
        steps = config["BO"].as_int("steps")
        print("BO will run for {} steps".format(steps))
        print("starting BO...")
        regret = None
        all_context, names = loop_methods(deepcopy(context), config, inspector)
        for i, cont in enumerate(all_context):
            method = method_names[cont.estimator.__class__.__name__]
            if method not in all_seed_regrets.keys():
                all_seed_regrets[method] = []
            output_path = None
            if base_path is not None:
                os.makedirs(base_path, exist_ok=True)
                output_path = "{}/{}/{}_steps_{}".format(base_path, names[i], steps, time.time())
                cont.inspector.set_inspector_path(output_path)
            print("starting {} estimator...".format(method))
            regret, all_regrets = run(cont, steps, function, output_path, seed)
            all_seed_regrets[method].append(regret[0])
            for step in range(len(all_regrets)):
                all_seed_regrets_per_step["method"].append(method)
                all_seed_regrets_per_step["step"].append(step)
                all_seed_regrets_per_step["seed"].append(seed)
                all_seed_regrets_per_step["regret"].append(all_regrets[step])
            print("Final Regret for seed {}: Final Regret: {}, All Regrets: {}".format(seed, regret[0], all_regrets))
            # f = open("{}/regrets.txt".format(base_path), "a")
            # f.write("{} - {}".format(names[i], regret))
            # f.close()
        # write system info
        original_stdout = sys.stdout  # Save a reference to the original standard output
        if base_path is not None:
            with open(base_path+'/sys_info.txt', 'w') as f:
                sys.stdout = f  # Change the standard output to the file we created.
                get_modules_info()
                get_system_info()
                get_cpu_info()
                sys.stdout = original_stdout  # Reset the standard output to its original value
    pd_final_regret = pd.DataFrame.from_dict(all_seed_regrets)
    print(pd_final_regret.describe())

    data = pd.DataFrame.from_dict(all_seed_regrets_per_step)
    print(data)
    sns.lineplot(data=data, x="step", y="regret", hue="method", estimator=np.mean, ci=95)
    plt.show()
