import numpy as np
from bayesian_optimization.utils.utils import config_list_float_or_none
import dill as pickle
import os
def get_new_samples(seed, config):
    np.random.seed(seed)
    function_set = config["BO"]["function"]
    start_samples = config["BO"]["n_train"]
    if seed in range(1,101) and str(start_samples) == str(8):
        loaded_samples = pickle.load(open(r"{}/sample_gen.pickle".format(os.path.dirname(__file__)), "rb"))
        if function_set in loaded_samples:
            samples_x = loaded_samples[function_set][str(seed)]
            return samples_x
    samples_x = np.random.uniform(
        low=config_list_float_or_none(config["BO"], "lower_bounds"),
        high=config_list_float_or_none(config["BO"], "upper_bounds"),
        size=(config["BO"].as_int("n_train"), len(config_list_float_or_none(config["BO"], "upper_bounds")))
    )
    return samples_x
