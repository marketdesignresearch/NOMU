from bayesian_optimization.plotter.plot import Plot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from bayesian_optimization.bo.bo import BO
from bayesian_optimization.acq_optimizer.gridsearch import GridSearch
import pandas as pd


class BOStep3DPlot(Plot):

    def __init__(self, figsize, title, out_path):
        super().__init__(figsize, title, out_path)
        self.legend = None
        self.data_dict = {}
        self.dat = {}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # Generate a contour plot

    def get_data_from(self, data):
        optimizer_x = data[GridSearch.INSP_X]
        for i in range(len(optimizer_x[0])):
            self.data_dict["x_{}".format(i)] = optimizer_x[:,i]
        self.data_dict["acqs"] = data[GridSearch.INSP_FINAL][GridSearch.INSP_ACQ]
        self.data_dict["samples_x"] = data[BO.INSP_SAMPLES_X]
        self.data_dict["samples_y"] = data[BO.INSP_SAMPLES_Y]
        self.data_dict["new_sample_x"] = data[BO.INSP_NEW_SAMPLE_X]
        self.data_dict["new_sample_y"] = data[BO.INSP_NEW_SAMPLE_Y]


        dat = {
            "x_0": [],
            "x_1": [],
            "acqs": [],
        }
        for i, x in enumerate(self.data_dict["x_0"]):
            dat["x_0"].append(x)
            dat["x_1"].append(self.data_dict["x_1"][i])
            dat["acqs"].append(self.data_dict["acqs"][i][0])
        self.dat = dat


    def drawAcqGridsearch(self, lb=None, ub=None):
        df = pd.DataFrame.from_dict(self.dat)
        sns.set_theme()
        Z = df.pivot_table(index='x_0', columns='x_1', values='acqs').T.values

        X_unique = np.sort(df.x_0.unique())
        Y_unique = np.sort(df.x_1.unique())
        X, Y = np.meshgrid(X_unique, Y_unique)


        # Generate a contour plot
        cbarticks = np.arange(-2., 2., 0.1)
        cbarlabels = np.around(np.linspace(-2., 2., num=11, endpoint=True), 2)
        cs = self.ax.contourf(X, Y, Z, cbarticks, vmin=-2., vmax=2., cmap="viridis")
        cb = self.fig.colorbar(cs, ticks=cbarticks)
        cb.set_ticks(cbarlabels)
        cb.set_ticklabels(cbarlabels)

        return self

    def drawFunctionOptima(self, function):
        optima = np.array(function.get_maxima_x())
        self.ax.scatter(optima[:,0], optima[:,1], c="blue", marker="^")
        return self

    def drawsamples(self):
        self.ax.scatter(self.data_dict["samples_x"][:8,0], self.data_dict["samples_x"][:8,1], c="black", marker="x")
        self.ax.scatter(self.data_dict["samples_x"][8:,0], self.data_dict["samples_x"][8:,1], c="black", marker="+")
        return self

    def drawNewsamples(self):
        self.ax.scatter([self.data_dict["new_sample_x"][0][0]], [self.data_dict["new_sample_x"][0][1]], c="red")
        return self
