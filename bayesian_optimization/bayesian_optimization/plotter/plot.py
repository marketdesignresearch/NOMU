import matplotlib.pyplot as plt
import seaborn as sns
import os

class Plot:

    def __init__(self, figsize, title, out_path):
        self.fig = plt.figure(figsize=figsize)
        self.title = title
        self.out_path = out_path

    def with_sns(self, rc={"lines.linewidth": 2.5}):
        sns.set_theme()
        sns.color_palette("deep")
        sns.set_context("paper", rc=rc, font_scale=1.7)
        sns.axes_style("whitegrid")
        return self

    def set_sns(self, **kwargs):
        sns.set(**kwargs)
        return self

    def show(self):
        plt.title(self.title)
        plt.tight_layout()
        plt.show()

    def save(self):
        plt.title(self.title)
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(self.out_path)

    def set_legend_outside(self, title, new_labels=None):
        leg = plt.legend(title=title, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        if new_labels is not None:
            for t, l in zip(leg.texts, new_labels): t.set_text(l)
        return self

    def set_legend_inside(self, title, new_labels=None, **kwargs):
        leg = plt.legend(title=title, loc=0, borderaxespad=0., **kwargs)
        for line in leg.get_lines():
            line.set_linewidth(3.0)
        if new_labels is not None:
            for t, l in zip(leg.texts, new_labels): t.set_text(l)
        return self

    def set_margin(self, x=0, y=0):
        axs = plt.gca()
        axs.margins(x=x,y=y)
        return self

    def set_axis_xlabel(self, label):
        axs = plt.gca()
        axs.set_xlabel(label)
        return self

    def set_axis_ylabel(self, label):
        axs = plt.gca()
        axs.set_ylabel(label)
        return self