#from __future__ import annotations
# Libs
from copy import deepcopy
import numpy as np
import h5py
import dill as pickle

# Internal
from bayesian_optimization.utils.utils import get_grid_points_multidimensional_and_grid
from bayesian_optimization.estimators import estimator

# Type hints
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from bayesian_optimization.estimators import estimator
    from configobj import ConfigObj


class Inspector:
    """Class used to store data from various processes. Main goal is to store temporary results as well as
    inspection data which then can be used to verify the processes.
    However this all can also be used to store further meta data such as time spent for certain calculations.
    """

    def __init__(
            self,
            lower_bounds: np.array,
            upper_bounds: np.array,
            n_test_points: int,
            inspector_path: str = None,
            do_estimate_test_data: bool = False,
            do_inspect_estimation: bool = False,
            do_inspect_optimization: bool = False,
            do_inspect_nn_model_fit: bool = False,
            do_inspect_acq: bool = False,
            store_estimators: bool = False,
    ):
        """constructor
        :param lower_bounds: lower bound for each dimension constraining space where to sample inspection data from
        :param upper_bounds: upper bound for each dimension constraining space where to sample inspection data from
        :param n_test_points: number of inspection points (per dimension)
        :param do_estimate_test_data: apply the estimation to the test data to get mu; sigma for inspection points.
        :param do_inspect_estimation: get inspection data for estimation
        :param do_inspect_optimization: get inspection data for optimization
        :param do_inspect_nn_model_fit: get inspection data for model fit
        """
        self.lower_bounds: np.array = lower_bounds
        self.upper_bounds: np.array = upper_bounds
        self.n_test_points: int = n_test_points
        self.inspector_path = inspector_path
        test_points, grid = get_grid_points_multidimensional_and_grid(self.lower_bounds, self.upper_bounds, self.n_test_points)
        self.test_x: np.array = test_points
        self.test_grid: np.array = grid
        self.test_y: Union[np.array, None] = None
        self.context: Union[Context, None] = None
        # Estimator
        self.estimate_test_data: bool = do_estimate_test_data
        self.inspect_estimation: bool = do_inspect_estimation
        self.store_estimators: bool = store_estimators
        self.estimators: np.array = []
        self.estimations: np.array = []
        self.estimations_on_test_data: np.array = []
        # ACQ Optimizer
        self.inspect_optimization: bool = do_inspect_optimization
        self.optimizations: np.array = []
        self.optimization_on_test_data: np.array = []
        # NNModel
        self.inspect_nn_model: bool = do_inspect_nn_model_fit
        self.fits: np.array = []
        # ACQ Func.
        self.inspect_acq: bool = do_inspect_acq
        self.acqs: np.array = []
        self.data_dump_dict = {}

    def reset_dump(self) -> NoReturn:
        """resets the dict which is used to flexibly store intermediate results
        :return:
        """
        self.data_dump_dict = {}

    def dump_data(self, key: str, data) -> NoReturn:
        """store the gicen data under the given key into one big dict.
        :param key: key for the dict
        :param data: data to store at under the key
        :return:
        """
        self.data_dump_dict[key] = data

    def save_dump_as_h5(self, path, filename):
        h5f = h5py.File("{}/{}.h5".format(path, filename), 'w')
        for key in self.data_dump_dict:
            h5f.create_dataset(key, data=self.data_dump_dict[key])
        h5f.close()

    def save_dump_as_pickle(self, path, filename):
        file_to_write = open("{}/{}.pickle".format(path, filename), "wb")
        pickle.dump(self.data_dump_dict, file_to_write)
        file_to_write.close()

    def set_context(self, context) -> NoReturn:
        """Set the context to the Inspector to allow access from both sides
        :param context: context
        """
        self.context = context
        self.test_y = self.context.callback(self.test_x)

    def set_inspector_path(self, path: str) -> NoReturn:
        self.inspector_path = path

    def get_json(self):
        json_dict = {}
        keys = ["lower_bounds", "upper_bounds", "n_test_points", "test_x", "test_y", "estimate_test_data",
                "inspect_estimation", "estimations", "estimations_on_test_data", "inspect_optimization",
                "optimizations", "optimization_on_test_data"]
        for k in keys:
            json_dict[k] = eval(getattr(self, k))
        return json_dict

    def add_estimation(self, estimation: dict) -> NoReturn:
        """Store Inspector data of one estimation of a estimator for the 'live' data the estimation is applied on
        :param estimation: dictionary containing information to be stored
        """
        self.estimations.append(estimation)

    def add_estimation_test_data(self, estimation: dict) -> NoReturn:
        """Store Inspector data of one estimation of a estimator for the test data specified in the Inspector
        :param estimation: dictionary containing information to be stored
        """
        self.estimations_on_test_data.append(estimation)

    # def reset_estimations(self):
    #    self.estimations = []

    def add_optimization(self, optimization: dict) -> NoReturn:
        """Store Inspector data of one optimization of a Optimizer for the the 'live' data the optimization
        is applied on.
        :param optimization: dictionary containing information to be stored
        """
        if len(self.optimizations)-1 >= self.context.bo_step:
            self.optimizations[self.context.bo_step] = optimization
        else:
            self.optimizations.append(optimization)

    def replace_last_optimization(self, optimization: dict) -> NoReturn:
        """Store Inspector data of one optimization of a Optimizer for the the 'live' data the optimization
        is applied on.
        :param optimization: dictionary containing information to be stored
        """
        self.optimizations.append(optimization)

    def add_fit(self, fit: dict) -> NoReturn:
        """Store Inspector data of one Keras model fit.
        :param fit: dictionary containing information to be stored
        """
        self.fits.append(fit)

    def add_acq_evaluation(self, acq_evaluation: dict) -> NoReturn:
        """Store Inspector data of a acquisition function evaluation.
        :param acq_evaluation: dictionary containing information to be stored
        """
        if len(self.acqs)-1 >= self.context.bo_step:
            self.acqs[self.context.bo_step] = acq_evaluation
        else:
            self.acqs.append(acq_evaluation)

    def replace_last_acq_evaluation(self, acq_evaluation: dict) -> NoReturn:
        """Store Inspector data of a acquisition function evaluation.
        :param acq_evaluation: dictionary containing information to be stored
        """
        self.acqs.pop()
        self.acqs.append(acq_evaluation)

    def add_estimator(self, estimator: estimator):
        self.estimators.append(deepcopy(estimator))

    @classmethod
    def read_inspector(cls, config: 'ConfigObj') -> 'Inspector':
        """reads the configuration from the config file and creates the inspector accordingly.
        :param config: config parser instance
        :return: inspector instance
        """
        lower_bounds = [float(i) for i in config["Inspector"].as_list("lower_bound")]
        upper_bounds = [float(i) for i in config["Inspector"].as_list("upper_bound")]

        return cls(
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            n_test_points=config["Inspector"].as_int("n_test_points"),
            do_estimate_test_data=config["Inspector"].as_bool("do_estimate_test_data"),
            do_inspect_estimation=config["Inspector"].as_bool("do_inspect_estimation"),
            do_inspect_optimization=config["Inspector"].as_bool("do_inspect_optimization"),
            do_inspect_acq=config["Inspector"].as_bool("do_inspect_acq"),
            store_estimators=config["Inspector"].as_bool("store_estimators"),
        )


