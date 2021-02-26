from bayesian_optimization.plotter.plot import Plot
from bayesian_optimization.datahandler.data_table import COLUMNS
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class RegretPlot(Plot):

    def __init__(self, figsize, title, out_path, df):
        super().__init__(figsize, title, out_path)
        self.df = df
        self.legend = None

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

    def draw_line_with_mean(self, ax=1, x=COLUMNS.STEP, y=COLUMNS.REGRET, hue=COLUMNS.METHODS, key_list=None, value_list=None, ci=None, linestyle="-", **kwargs):
        return self.draw_line(ax=ax, x=x, y=y, hue=hue, key_list=key_list, value_list=value_list, ci=ci, estimator=np.mean, linestyle=linestyle, **kwargs)

    def draw_line_with_median(self, ax=1, x=COLUMNS.STEP, y=COLUMNS.REGRET, hue=COLUMNS.METHODS, key_list=None, value_list=None, ci=None,linestyle="-", **kwargs):
        return self.draw_line(ax=ax, x=x, y=y, hue=hue, key_list=key_list, value_list=value_list, ci=ci, estimator=np.median, linestyle=linestyle, **kwargs)

    def draw_line(
            self,
            ax=1,
            x=COLUMNS.STEP,
            y=COLUMNS.REGRET,
            hue=COLUMNS.METHODS,
            key_list=None,
            value_list=None,
            ci=None,
            estimator=np.mean,
            linestyle="-",
            **kwargs
    ):
        if key_list is None:  #
            key_list = [COLUMNS.METHODS]
        if value_list is None:  #
            value_list = [("GP",), ("NOMU",), ("DE",), ("DO",)]
        data = self.df[self.df[key_list].apply(tuple, axis=1).isin(value_list)]
        dashes = [(1, 0) for i in data[hue].unique()]
        if linestyle == "-":
            dashes = [(1,0) for i in data[hue].unique()]
        if linestyle == "--":
            dashes = [(2,1) for i in data[hue].unique()]
        splot = sns.lineplot(data=data, x=x, y=y, hue=hue, estimator=estimator, ci=ci, dashes=dashes, style=hue, **kwargs)
        return self







