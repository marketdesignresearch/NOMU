from enum import Enum
import numpy as np
import glob
import os
import dill as pickle
from tensorflow.keras import backend as K
import pandas as pd

# Type hinting
from typing import *
from typing import NoReturn

if TYPE_CHECKING:
    from bayesian_optimization.context.context import Context
    from bayesian_optimization.functions.abstract_function import AbstractFunction

class COLUMNS(str, Enum):
    METHODS = "methods"
    SCALE = "scale"
    REGRET = "regret"
    END_REGRET = "end_regret"
    STEP = "step"
    RUN = "run"
    ACQ = "acq"
    OPTIMIZER_CALLBACK = "opt_callback"
    FUNCTION = "function"
    C = "c_factor"
    C_STEP = "c_step"

class BODataTable():
    """class that handles the data from the context files saved as pickles.
    The pickle fies can be read by this class. All important data is extracted from the context files
    and saves as a datatable (dataframe) from pandas.

    """

    METHODS = {
        "SingleMethod": "UB",
        "SingleMethodBounded": "NOMU",
        "SampleMethod": "DO",
        "GP": "GP",
        "EnsembleMethod": "DE",
        "HyperDeepEnsembleMethod": "HDE",
    }

    ACQS = {
        "UpperBound": "UB",
        "ExpectedImprovement": "EI"
    }


    OPT_CB = {
        "DynamicC": "DC_lin",
        "DynamicCExponential": "DC_exp"
    }

    OPT_CB_NAME = {
        "DC_lin": "Dynamic C linear",
        "DC_exp": "Dynamic C exponential",
        "No_DC": "No dynamic C"
    }

    def __init__(self, columns: np.array):
        self.data_dict = {}
        self.columns = columns
        for c in columns:
            self.data_dict[c] = []

    def _get_regret_of_samples(self, context: 'Context', step: int, y_opt: float) -> float:
        """calculates the regrets based on the samples taken until this step.
        :param context: context to analyze
        :param step: step to analyze
        :param y_opt: true optima of the target function
        :return: minimal regret over all samples
        """
        initial_pts = len(context.samples_y)-context.bo_step-1
        y_samp = context.samples_y
        if isinstance(y_samp[0], float):
            y_samp = np.array([[y] for y in y_samp])
        if not isinstance(y_opt, float):

            return np.abs(np.max(y_samp[0:initial_pts+step, 0]) - y_opt[0])
        return np.abs(np.max(y_samp[0:initial_pts+step]) - y_opt)

    def _get_regret_of_new_arg_max(self, context: 'Context', step: int, y_opt: float, data: pd.DataFrame, function: 'AbstractFunction'):
        """calculates the regret based on the new proposed sample

        :param context: context to analyze
        :param step: step to analyze
        :param y_opt: true optima of the target function
        :param data: dataframe
        :param function: target function
        :return: minimal regret over all samples
        """
        y_evaluation = function.evaluate_scaled([data[context.acq_optimizer.INSP_FINAL][context.acq_optimizer.INSP_ARG_MAX]])[0]
        if not isinstance(y_evaluation, float):
            return np.abs(y_evaluation[0] - y_opt[0])
        return np.abs(y_evaluation - y_opt)

    def _get_method(self, context: 'Context') -> NoReturn:
        """read the used estimator method from the context and save it into the data dict

        :param context: context to analyze
        :return:
        """
        self.data_dict[COLUMNS.METHODS].append(self.METHODS[context.estimator.__class__.__name__])

    def _get_scale(self, context: 'Context') -> NoReturn:
        """read the used mean width budget from the context and save it into the data dict

        :param context: context to analyze
        :return:
        """
        acq = context.acq
        for d in range(0, 10):
            # only check nesting depth 10
            if (acq.__class__.__name__ != "MeanWidthScaled" and acq.__class__.__name__ != "MeanWidthScaledMC") and hasattr(acq, 'acq_to_decorate'):
                acq = acq.acq_to_decorate
            else:
                break
        if hasattr(acq, 'scale_mean_width') and acq.scale_mean_width:
            self.data_dict[COLUMNS.SCALE].append(acq.scale_mean_width)
        else:
            self.data_dict[COLUMNS.SCALE].append("None")

    def _get_acq(self, context: 'Context') -> NoReturn:
        """read the used acquisition function from the context and save it into the data dict

        :param context: context to analyze
        :return:
        """
        acq = context.acq
        for i in range(0,10):
            if hasattr(acq, "acq_to_decorate"):
                acq = acq.acq_to_decorate
            else:
                break
        self.data_dict[COLUMNS.ACQ].append(self.ACQS[acq.__class__.__name__])

    def _get_opt_cb(self, context: 'Context') -> NoReturn:
        """read the used acquisition function optimizer callback from the context and save it into the data dict

        :param context: context to analyze
        :return:
        """
        acq_opt = context.acq_optimizer
        if acq_opt.callback is not None:
            self.data_dict[COLUMNS.OPTIMIZER_CALLBACK].append(self.OPT_CB[acq_opt.callback.__class__.__name__])
        else:
            self.data_dict[COLUMNS.OPTIMIZER_CALLBACK].append("No_DC")

    def _get_regret(self, context: 'Context', step: int, data: pd.DataFrame, function: AbstractFunction) -> NoReturn:
        """calls the functions to get the regret of the samples and of the proposed samples
        and save it to the data dict

        :param context: context to analyze
        :param step: step to analyze
        :param data: dataframe
        :param function: target function
        :return:
        """
        y_opt = function.evaluate_scaled(function.get_maxima_x())[0]
        self.data_dict[COLUMNS.REGRET].append(
            np.min([self._get_regret_of_new_arg_max(context, step, y_opt, data, function), self._get_regret_of_samples(context, step, y_opt)]))

    def _get_regret_last(
            self,
            context: 'Context',
            step: int,
            data: pd.DataFrame,
            function: AbstractFunction,
            tot_steps: int
    ) -> NoReturn:
        """get the regret after the last step and store it in the data dict

        :param context: context to analyze
        :param step: step to analyze
        :param data: dataframe
        :param function: target function
        :param tot_steps: total number of Bo-steps assigned to the BO run
        :return:
        """
        y_opt = function.evaluate_scaled(function.get_maxima_x())[0]
        self.data_dict[COLUMNS.END_REGRET].append(
            np.min([self._get_regret_of_new_arg_max(context, step, y_opt, data, function), self._get_regret_of_samples(context, tot_steps, y_opt)]))

    def _get_initial_regret(
            self,
            context: 'Context',
            function: AbstractFunction,
    ) -> NoReturn:
        """ get the regret only from the starting samples

        :param context: context to analyze
        :param function: target function
        :return:
        """
        y_opt = function.evaluate_scaled(function.get_maxima_x())[0]
        self.data_dict[COLUMNS.REGRET].append(self._get_regret_of_samples(context, 0, y_opt))

    def _get_initial_regret_last(
            self,
            context: 'Context',
            function: AbstractFunction,
            data: pd.DataFrame,
            tot_steps: int
    ) -> NoReturn:
        """get the regret only from the starting samples for last step

        :param context: context to analyze
        :param function: target function
        :param data: dataframe
        :param tot_steps: total number of Bo-steps assigned to the BO run
        :return:
        """
        y_opt = function.evaluate_scaled(function.get_maxima_x())[0]
        self.data_dict[COLUMNS.END_REGRET].append(self._get_regret_of_samples(context, tot_steps, y_opt))

    def _get_c(
            self,
            context: 'Context',
            data: pd.DataFrame
    ) -> NoReturn:
        """extract the c values from the contexts

        :param context: context to analyze
        :param data: dataframe
        :return:
        """
        if context.acq_optimizer.callback is not None:
            dyn_steps = data[context.acq_optimizer.CALLBACK]
        else:
            dyn_steps = []
        if len(dyn_steps) == 0:
            if hasattr(context.acq, "INSP_C"):
                self.data_dict[COLUMNS.C].append(data[context.acq.INSP_C])
                self.data_dict[COLUMNS.C_STEP].append(0)
            else:
                self.data_dict[COLUMNS.C].append(1.0)
                self.data_dict[COLUMNS.C_STEP].append(0)
        else:
            if "c" in dyn_steps[-1].keys():
                self.data_dict[COLUMNS.C].append(dyn_steps[-1]["c"])
            else:
                self.data_dict[COLUMNS.C].append(dyn_steps[-1]["opt_acq"])

            self.data_dict[COLUMNS.C_STEP].append(len(dyn_steps))

    def add_combination(self, combi: np.array, name: str) -> 'BODataTable':
        """adds an additional column to the dataframe containing containing a label
        consisting of column values concatenated and a prepend name

        :param combi: columns to concatenate as an array, column-name from COLUMNS
        :param name: name to prepend
        :return:
        """
        dict_key = "_".join(combi)
        dict_key = "_".join([dict_key, name])
        self.data_dict[dict_key] = []

        for i, val in enumerate(self.data_dict[combi[0]]):
            label = val
            for key in combi[1:]:
                label = "{}_{}".format(label, self.data_dict[key][i])
            label = "{}_{}".format(label, name)
            self.data_dict[dict_key].append(label)
        return self

    def get_data_from(
            self,
            path_to_runs: str,
            function: AbstractFunction,
            overwrite_method:str=None
    ) -> NoReturn:
        """read the data from the context-pickel-file located at the give path

        :param path_to_runs: path to the context-file
        :param function: target function
        :param overwrite_method: alternative methods name
        :return:
        """
        for run in glob.glob(r"{}/**/context.pickle".format(path_to_runs), recursive=True):
            run_name = run.split(path_to_runs)[1].split(os.sep)[1].split("_")[0]
            tot_steps = len(glob.glob(os.path.dirname(run)+r"/BO_steps/*.pickle"))
            with open(run, "rb") as f:
                context = pickle.load(f)
                context.nn_model = None
                context.model_optimizer = None
                context._model_optimizer_config = None
                f.close()
                K.clear_session()
            with open(glob.glob(os.path.dirname(run)+r"/BO_steps/*.pickle")[-1], "rb") as f:
                data_last = pickle.load(f)
                f.close()
            if COLUMNS.METHODS in self.columns:
                if overwrite_method:
                    self.data_dict[COLUMNS.METHODS].append(overwrite_method)
                else:
                    self._get_method(context)
            if COLUMNS.SCALE in self.columns:
                self._get_scale(context)
            if COLUMNS.REGRET in self.columns:
                self._get_initial_regret(context, function)
            if COLUMNS.END_REGRET in self.columns:
                self._get_initial_regret_last(context, function, data_last, tot_steps)
            if COLUMNS.ACQ in self.columns:
                self._get_acq(context)
            if COLUMNS.STEP in self.columns:
                self.data_dict[COLUMNS.STEP].append(0)
            if COLUMNS.FUNCTION in self.columns:
                self.data_dict[COLUMNS.RUN].append(function.__class__.__name__)
            if COLUMNS.RUN in self.columns:
                self.data_dict[COLUMNS.RUN].append(run_name)
            if COLUMNS.OPTIMIZER_CALLBACK in self.columns:
                self._get_opt_cb(context)
            if COLUMNS.C in self.columns:
                self.data_dict[COLUMNS.C].append(0)
                self.data_dict[COLUMNS.C_STEP].append(0)
            for i, step in enumerate(glob.glob(os.path.dirname(run)+r"/BO_steps/*.pickle")):
                st = int(step.split(os.sep)[-1].split(".pickle")[0])
                with open(step, "rb") as f:
                    data = pickle.load(f)
                    f.close()
                if COLUMNS.METHODS in self.columns:
                    if overwrite_method:
                        self.data_dict[COLUMNS.METHODS].append(overwrite_method)
                    else:
                        self._get_method(context)
                if COLUMNS.SCALE in self.columns:
                    self._get_scale(context)
                if COLUMNS.REGRET in self.columns:
                    self._get_regret(context, st, data, function)
                if COLUMNS.END_REGRET in self.columns:
                    self._get_regret_last(context, st, data_last, function, tot_steps)
                if COLUMNS.ACQ in self.columns:
                    self._get_acq(context)
                if COLUMNS.STEP in self.columns:
                    self.data_dict[COLUMNS.STEP].append(st+1)
                if COLUMNS.RUN in self.columns:
                    self.data_dict[COLUMNS.RUN].append(run_name)
                if COLUMNS.FUNCTION in self.columns:
                    self.data_dict[COLUMNS.RUN].append(function.__class__.__name__)
                if COLUMNS.OPTIMIZER_CALLBACK in self.columns:
                    self._get_opt_cb(context)
                if COLUMNS.C in self.columns:
                    self._get_c(context, data)

    def get_best(
            self,
            target_cat: str,
            target: str,
            category: np.array,
            key_list: np.array,
            value_list: np.array,
            modifier: str
    ):
        """get the "best" value of the target category  of the specified category
        'best' meaning the minimum mean/median value
        :param target_cat: which category should be returned
        :param target: for which category the minimum mean should be considered
        :param category: get the best for different categories, specify which
        :param key_list: list of keys to filer the data
        :param value_list: value for the filter-keys
        :param modifier: get the mean or de median
        :return: 
        """
        df = pd.DataFrame.from_dict(self.data_dict)
        data = df[df[key_list].apply(tuple, axis=1).isin(value_list)]
        group_list = list(category)
        group_list.append(target_cat)
        data_grouped = data.groupby(group_list, as_index=False)
        if modifier == "mean":
            mean_gr = data_grouped.mean().reset_index()
            return mean_gr.loc[mean_gr.groupby(category)[target].idxmin()]
        if modifier == "median":
            median_gr = data_grouped.median().reset_index()
            return median_gr.loc[median_gr.groupby(category)[target].idxmin()]