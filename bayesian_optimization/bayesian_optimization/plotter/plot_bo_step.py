import matplotlib.pyplot as plt
from bayesian_optimization.plotter.bo_step_2d_plot import BOStep2DPlot

def run(context, resolution, step, path, out_path,lower_bound=[-1.], upper_bound=[1.]):
    plot = BOStep2DPlot((12,7),"Step {}".format(step), "{}/step_{}_plot.png".format(out_path,step), context, 2000, step)
    plot.prepare()
    plot.with_sns()\
        .with_axis_range(1, (-2,2))\
        .with_axis_range(2, (0,4))\
        .draw_true_f()\
        .draw_samples()\
        .draw_sigma()\
        .draw_acq()\
        .draw_mu()\
        .draw_acq_max()\
        .draw_uncertainty_bounds()\
        .draw_sample_vert_line()\
        .set_axis_ylabel(1, "True Function & Mean")\
        .set_axis_ylabel(2, "Acquisition Function & Sigma")\
        .save()
    plt.clf()

