from bayesian_optimization.plotter.plot import Plot

class Plot2D(Plot):

    def __init__(self, figsize, title, out_path):
        super().__init__(figsize, title, out_path)

    def init(self):
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.margins(x=0)
        self.axis = []
        self.axis.append(self.ax1)
        return self

    def _add_twinx_if_needed(self, number: int):
        if len(self.axis) < number:
            ax = self.ax1.twinx()
            ax.margins(x=0)
            self.axis.append(ax)



