from bayesian_optimization.plotter.plot_2d import Plot2D
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import Rectangle
import numpy as np
from bayesian_optimization.bo.bo import BO

class BOStep2DPlot(Plot2D):

    def __init__(self, figsize, title, out_path, data, context, function):
        super().__init__(figsize, title, out_path)
        self.function = function
        self.data = data
        self.context = context
        self.test_x = self.data[context.acq_optimizer.INSP_X]
        self.mu = self.data[context.acq_optimizer.INSP_FINAL][context.acq_optimizer.INSP_MU]
        self.sig = self.data[context.acq_optimizer.INSP_FINAL][context.acq_optimizer.INSP_SIGMA]
        self.acq = self.data[context.acq_optimizer.INSP_FINAL][context.acq_optimizer.INSP_ACQ]
        self.samples_x = self.data[BO.INSP_SAMPLES_X]
        self.samples_y = self.data[BO.INSP_SAMPLES_Y]
        self.new_sample_x = self.data[BO.INSP_NEW_SAMPLE_X]
        self.new_sample_y = self.data[BO.INSP_NEW_SAMPLE_Y]
        self.step = self.context.bo_step
        self.legend_handles = []
        self.legend_labels = []
        if hasattr(context.acq, "INSP_C"):
            self.mws_c = self.data[context.acq.INSP_C]

    def with_sns(self, rc={"lines.linewidth": 2.5, "axes.facecolor": ".9"}):
        #sns.set_theme()
        sns.set_style("ticks")

        sns.color_palette("deep")
        sns.set_context("paper", rc=rc, font_scale=1.7)
        #sns.axes_style("darkgrid", {"axes.facecolor": ".9"})
        return self

    def with_axis_range(self, index:int, ylim):
        self._add_twinx_if_needed(index)
        self.axis[index-1].set_ylim(ylim)
        return self

    def draw_true_f(self, ax=1, linestyle="-", color="black", label="True Function", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        y_true = [y[0] for y in self.function.evaluate_scaled(self.test_x)]
        h = self.axis[ax-1].plot(self.test_x, y_true, linestyle=linestyle, color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_uncertainty_bounds(self, ax=1, color="C0", alpha=0.3, label="Uncertainty Bounds", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].fill_between(self.test_x[:,0], self.mu[:,0] - self.sig[:,0]*self.mws_c, self.mu[:,0] + self.sig[:,0]*self.mws_c, color=color, alpha=alpha, **kwargs)
        if in_legend:
            self.legend_handles.append(h)
            self.legend_labels.append(label)
        return self

    def draw_mu(self, ax=1, color="C0", label="Mean estimation", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].plot(self.test_x, self.mu[:,0], color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_acq(self, ax=2, color="C1", label="Acquisition Function", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].plot(self.test_x, self.acq, color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_acq_area(self, ax=2, color="C1", label="Acquisition Function", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].fill_between(self.test_x[:,0], np.full_like(self.acq, 0)[:,0], self.acq[:,0], color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h)
            self.legend_labels.append(label)
        return self

    def draw_samples(self, ax=1, hightlight_last=True, marker="o", s=80., hightlight_color="red", color="black", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        self.axis[ax-1].scatter(self.samples_x[:-1], self.samples_y[:-1], marker=marker, s=s, color=color, **kwargs)
        if self.step == 0 or not hightlight_last:
            h = self.axis[ax-1].scatter(self.samples_x, self.samples_y, marker=marker, s=s, color=color, **kwargs)
            if in_legend:
                self.legend_handles.append(h)
                self.legend_labels.append("Samples")
        else:
            h1 = self.axis[ax-1].scatter(self.samples_x[:-1], self.samples_y[:-1], marker=marker, s=s, color=color, **kwargs)
            h2 = self.axis[ax-1].scatter(self.samples_x[-1:], self.samples_y[-1:], marker=marker, s=s, color=hightlight_color, **kwargs)
            if in_legend:
                self.legend_handles.append(h1)
                self.legend_handles.append(h2)
                self.legend_labels.append("Samples")
                self.legend_labels.append("New Sample")
        return self

    def draw_sigma(self, ax=2, color="C2", linestyle="--", label="Sigma", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].plot(self.test_x, self.sig*self.mws_c, linestyle=linestyle, color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_sigma_orig(self, ax=2, color="C3", linestyle="--", label="Sigma not scaled", in_legend=True, **kwargs):
        self._add_twinx_if_needed(ax)
        h = self.axis[ax-1].plot(self.test_x, self.sig, linestyle=linestyle, color=color, **kwargs)
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_acq_max(self, ax=2, label="Acquisition Maxima", in_legend=True):
        self._add_twinx_if_needed(ax)
        max_x = self.data[self.context.acq_optimizer.INSP_FINAL][self.context.acq_optimizer.INSP_ARG_MAX]
        max_acq = self.data[self.context.acq_optimizer.INSP_FINAL][self.context.acq_optimizer.INSP_ACQ_ARG_MAX]
        h = self.axis[ax-1].plot(max_x, max_acq, "v", ms=10.0, color="red")
        if in_legend:
            self.legend_handles.append(h[0])
            self.legend_labels.append(label)
        return self

    def draw_sample_vert_line(self, ax=1):
        for x in self.samples_x:
            self.axis[ax-1].axvline(x=x, linewidth=1.5, color="gray", alpha=1.0)
        return self

    def set_axis_ylabel(self, ax, label):
        self.axis[ax-1].set_ylabel(label,labelpad=30)
        return self

    def add_mws_c(self):
        if self.context.acq_optimizer.callback is not None:
            dyn_steps = self.data[self.context.acq_optimizer.CALLBACK]
        else:
            dyn_steps = []
        if len(dyn_steps) == 0:
            self.text_to_legend("$c_{mws} = $" + str(round(self.data[self.context.acq.INSP_C], 5)))
        else:
            self.text_to_legend("$c_{mws} = $" + str(round(dyn_steps[0]["opt_acq"], 5)))
        return self

    def add_dc_c(self):
        if self.context.acq_optimizer.callback is not None:
            dyn_steps = self.data[self.context.acq_optimizer.CALLBACK]
            if len(dyn_steps) == 0:
                self.text_to_legend("$c_{dc} = $" + str(round(self.data[self.context.acq.INSP_C], 5)))
            else:
                self.text_to_legend("$c_{dc} = $" + str(round(dyn_steps[-1]["opt_acq"], 5)))
        return self


    def text_to_legend(self, label):
        extra = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
        self.legend_handles.append(extra)
        self.legend_labels.append(label)

    def set_legend_outside(self, title):
        leg = plt.legend(self.legend_handles, self.legend_labels, title=title, bbox_to_anchor=(1.15, 1), loc=2, borderaxespad=0.)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        return self

    def set_sns(self, **kwargs):
        sns.set(**kwargs)
        return self


    def show(self):
        plt.tight_layout()
        plt.title(self.title)
        plt.show()

    def save(self):
        plt.tight_layout()
        #sns.set_context(rc={"axes.facecolor": ".9"})
        self.ax1.tick_params(axis="x",direction="out", pad=22, which='both', length=10)
        plt.title(self.title)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        plt.savefig(self.out_path)

