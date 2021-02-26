from bayesian_optimization.plotter.plot import Plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import glob
import dill as pickle


class RegretPlot(Plot):

    METHODS = {
        "SingleMethod": "UB",
        "SingleMethodBounded": "NOMU",
        "SampleMethod": "DO",
        "GP": "GP",
        "EnsembleMethod": "DE",
    }

    def __init__(self, figsize, title, out_path, contexts, function):
        super().__init__(figsize, title, out_path)
        self.contexts = contexts
        self.data_dict = {
            "method": [],
            "scale": [],
            "regret": [],
            "step": [],
            "run": [],
            "acq": [],
        }
        self.function = function
        self.legend = None
        #self.test_points = get_grid_points_multidimensional(lower_bound, upper_bound, resolution)
        #self.test_x = self.test_points[:,0]
        #self.step = step
        #self.mu = None
        #self.sig = None
        #self.acq = None


    def _get_regret_of_samples(self, context, step, y_opt):
        initial_pts = len(context.samples_y)-context.bo_step-1
        return np.abs(np.max(context.samples_y[0:initial_pts+step]) - y_opt)

    def _get_regret_of_new_arg_max(self, context, step, y_opt, data):
        y_evaluation = context.callback([data[context.acq_optimizer.INSP_FINAL][context.acq_optimizer.INSP_ARG_MAX]])[0][0]
        return np.abs(y_evaluation - y_opt)

    def _get_scale(self, context):
        acq = context.acq
        for d in range(0, 10):
            # only check nesting depth 10
            if acq.__class__.__name__ != "MeanWidthScaled":
                acq = acq.acq_to_decorate
            else:
                break
        if acq.scale_mean_width:
            self.data_dict["scale"].append(acq.scale_mean_width)
        else:
            self.data_dict["scale"].append("None")

    def _get_method(self, context):
        self.data_dict["method"].append(self.METHODS[context.estimator.__class__.__name__])

    def _get_acq(self, context):
        acq = context.acq
        for i in range(0,10):
            if hasattr(acq, "acq_to_decorate"):
                acq = acq.acq_to_decorate
            else:
                break
        self.data_dict["acq"].append(acq.__class__.__name__)


    def _get_regret(self, context, step, data):
        y_opt = context.callback([np.array(self.function.get_optima_new())])[0][0]
        self.data_dict["regret"].append(
            np.min([self._get_regret_of_new_arg_max(context, step, y_opt, data), self._get_regret_of_samples(context, step, y_opt)]))

    def _get_initial_regret(self, context):
        y_opt = context.callback([np.array(self.function.get_optima_new())])[0][0]
        self.data_dict["regret"].append(self._get_regret_of_samples(context, 0, y_opt))

    def add_combination(self, combi):
        dict_key = "_".join(combi)
        self.data_dict[dict_key] = []
        for i, val in enumerate(self.data_dict[combi[0]]):
            label = val
            for key in combi[1:]:
                if key.startswith("["):
                    label = "{}_{}".format(label, key)
                else:
                    label = "{}_{}".format(label, self.data_dict[key][i])
            self.data_dict[dict_key].append(label)
        return self

    def prepare(self, exclude_runs=None):
        if exclude_runs == None:
            exclude_runs = []
        # step 0
        for context in self.contexts:
            run_path = os.path.basename(context.inspector.inspector_path.split("/scale")[0])
            skip = False
            for blocker in exclude_runs:
                if blocker in run_path:
                    skip = True
                    break
            if skip:
                continue
            self._get_method(context)
            self._get_scale(context)
            self._get_initial_regret(context)
            self._get_acq(context)
            self.data_dict["step"].append(0)
            self.data_dict["run"].append(run_path[0:6])
            for i, step in enumerate(glob.glob(context.inspector.inspector_path+"\\BO_steps\\*.pickle")):
                with open(step, "rb") as f:
                    data = pickle.load(f)
                    f.close()
                self._get_method(context)
                self._get_scale(context)
                self._get_regret(context, i, data)
                self._get_acq(context)
                self.data_dict["step"].append(i+1)
                self.data_dict["run"].append(run_path[0:6])
        return self



    def with_axis_range(self, index:int, ylim, visible=True):
        self._add_twinx_if_needed(index)
        self.axis[index-1].set_ylim(ylim)
        self.axis[index-1].patch.set_alpha(0.1)
        if not visible:
            self.axis[index-1].tick_params(
                axis='y',
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelright=False
            )
        return self


    def with_log_y(self):
        axs = plt.gca()
        axs.set(yscale="log")
        return self

    def set_y_lim(self, lim):
        axs = plt.gca()
        axs.set_ylim(lim)
        return self

    def draw_line_with_mean(self, ax=1, x="step", y="regret", hue="method", key_list=None, value_list=None, ci=None, linestyle="-", **kwargs):
        return self.draw_line(ax=ax, x=x, y=y, hue=hue, key_list=key_list, value_list=value_list, ci=ci, estimator=np.mean, linestyle=linestyle, **kwargs)

    def draw_line_with_median(self, ax=1, x="step", y="regret", hue="method", key_list=None, value_list=None, ci=None,linestyle="-", **kwargs):
        return self.draw_line(ax=ax, x=x, y=y, hue=hue, key_list=key_list, value_list=value_list, ci=ci, estimator=np.median, linestyle=linestyle, **kwargs)

    def draw_line(
            self,
            ax=1,
            x="step",
            y="regret",
            hue="method",
            key_list=None,
            value_list=None,
            ci=None,
            estimator=np.mean,
            linestyle="-",
            **kwargs
    ):
        if key_list is None:  #
            key_list = ["method"]
        if value_list is None:  #
            value_list = [("GP",), ("UB",), ("DE",), ("DO",)]
        df = pd.DataFrame.from_dict(self.data_dict)
        data = df[df[key_list].apply(tuple, axis=1).isin(value_list)]
        dashes = [(1, 0) for i in data[hue].unique()]
        if linestyle == "-":
            dashes = [(1,0) for i in data[hue].unique()]
        if linestyle == "--":
            dashes = [(2,1) for i in data[hue].unique()]
        splot = sns.lineplot(data=data, x=x, y=y, hue=hue, estimator=estimator, ci=ci, dashes=dashes, style=hue, **kwargs)
        return self







