import matplotlib.pyplot as plt
import numpy as np
MARKER_SIZE = 6
LINE_WIDTH = 2
TRANSPARENCY = 0.2


def plot_prediction(context, path, title="title", step=-1):

    fig = plt.figure(figsize=(16, 9))
    samples_x = context.estimator.get_inspector_samples_x_on_test_data(context, step)
    samples_y = context.estimator.get_inspector_samples_y_on_test_data(context, step)
    mus = context.estimator.get_inspector_mu_on_test_data(context, step)
    sigmas = context.estimator.get_inspector_sigma_on_test_data(context, step)
    acqs = context.estimator.get_inspector_acq_on_test_data(context,step)
    factor = context.inspector.acqs[step]["factor"]
    acq_new = context.acq.evaluate(mus, factor*sigmas, 1.0, False)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(context.inspector.test_x, context.inspector.test_y, label='True Function', color='black')
    plot_nnub = plt.plot(context.inspector.test_x, mus, linewidth=LINE_WIDTH, label='mean', linestyle='-')
    plt.plot(context.inspector.test_x, sigmas, color="red", linewidth=LINE_WIDTH, label='sigma', linestyle=':')
    plt.plot(context.inspector.test_x, acq_new, color="orange", linewidth=LINE_WIDTH, label='acquisition func.', linestyle='--')
    plt.fill_between(
        context.inspector.test_x[:,0],
        (mus[:,0]-(sigmas[:,0])),
        (mus[:,0]+(sigmas[:,0])),
        alpha=TRANSPARENCY, ec='None', label='NN Uncertainty Bounds')
    plt.plot(samples_x, samples_y, 'ko', markersize=MARKER_SIZE, label='Training Data')
    arg_max = context.acq_optimizer.get_inspector_arg_max(context, step)
    max_acq = context.acq_optimizer.get_inspector_acq_arg_max(context, step)
    mu_arg_max = context.acq_optimizer.get_inspector_mu_arg_max(context, step)
    sigma_arg_max = context.acq_optimizer.get_inspector_sigma_arg_max(context, step)
    plt.plot(arg_max, max_acq, 'ko', color="tab:red", markersize=MARKER_SIZE, label='Next Evaluation')

    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.ylim(-2., 2.)

    plt.grid(True)
    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig(path)
    plt.clf()


def plot_prediction3D(context, path, title="title", step=-1):

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection="3d")

    grid = context.inspector.test_grid
    ax.plot_surface(
        grid[0],
        grid[1],
        np.reshape(context.inspector.test_y, grid[0].shape),
        label='Goldstein Price',
        alpha=0.3
    )



    samples_x = context.estimator.get_inspector_samples_x_on_test_data(context, step)
    samples_y = context.estimator.get_inspector_samples_y_on_test_data(context, step)
    mus = context.estimator.get_inspector_mu_on_test_data(context, step)
    sigmas = context.estimator.get_inspector_sigma_on_test_data(context, step)
    acqs = context.estimator.get_inspector_acq_on_test_data(context,step)
    ax.plot_surface(
        grid[0],
        grid[1],
        np.reshape([a for a in acqs], grid[0].shape),
        label='Goldstein Price',
        alpha=0.3
    )

    ax.scatter(samples_x[:,0],samples_x[:,1], samples_y, 'ko', label='Training Data')

    arg_max = context.acq_optimizer.get_inspector_arg_max(context, step)
    max_acq = context.acq_optimizer.get_inspector_acq_arg_max(context, step)
    ax.scatter(arg_max[0], arg_max[1], max_acq, 'ko', color="tab:red", label='Next Evaluation')

    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig(path)
    plt.clf()


def plot_prediction3D_optima_axis_0(context, path, optima, title="title", step=-1):


    grid = context.inspector.test_grid
    samples_x = context.estimator.get_inspector_samples_x_on_test_data(context, step)
    samples_y = context.estimator.get_inspector_samples_y_on_test_data(context, step)
    mus = context.estimator.get_inspector_mu_on_test_data(context, step)
    sigmas = context.estimator.get_inspector_sigma_on_test_data(context, step)
    acqs = context.estimator.get_inspector_acq_on_test_data(context, step)

    idx = np.array([np.abs(x - optima[0][0]) for x in grid[0][0]]).argmin()
    x_ax = grid[1][:, 0]
    y_ax = np.reshape([a for a in acqs], grid[0].shape)[:, idx]
    y_true = np.reshape(context.inspector.test_y, grid[0].shape)[:, idx]


    fig = plt.figure(figsize=(16, 9))
    plt.plot(x_ax, y_true, label='True Function at optimal x_0',  color="blue")
    plt.plot(x_ax, y_ax, label='Acq Function',  color="orange")
    plt.plot(samples_x[:,0], samples_y, 'ko', label='Training Data smashed')

    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig(path)
    plt.clf()

def plot_prediction3D_optima_axis_1(context, path, optima, title="title", step=-1):
    grid = context.inspector.test_grid
    samples_x = context.estimator.get_inspector_samples_x_on_test_data(context, step)
    samples_y = context.estimator.get_inspector_samples_y_on_test_data(context, step)
    mus = context.estimator.get_inspector_mu_on_test_data(context, step)
    sigmas = context.estimator.get_inspector_sigma_on_test_data(context, step)
    acqs = context.estimator.get_inspector_acq_on_test_data(context, step)
    idx = np.array([np.abs(x - optima[0][1]) for x in grid[1][:, 0]]).argmin()
    grid = context.inspector.test_grid
    x_ax = grid[0][0]
    y_ax = np.reshape([a for a in acqs], grid[1].shape)[:, idx]

    y_true = np.reshape(context.inspector.test_y, grid[1].shape)[idx, :]

    fig = plt.figure(figsize=(16, 9))

    plt.plot(x_ax, y_true, label='True Function at optimal x_1', color="blue")
    plt.plot(x_ax, y_ax, label='Acq Function', color="orange")
    plt.plot(samples_x[:, 1], samples_y, 'ko', label='Training Data smashed')

    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig(path)
    plt.clf()

def plot_DE_individual_models(context, path, title="DE individual models", step=-1):
    fig = plt.figure(figsize=(16, 9))
    # overall results
    samples_x = context.estimator.get_inspector_samples_x_on_test_data(context, step)
    samples_y = context.estimator.get_inspector_samples_y_on_test_data(context, step)
    mu = context.estimator.get_inspector_mu_on_test_data(context, step)
    sigma = context.estimator.get_inspector_sigma_on_test_data(context, step)
    acq = context.estimator.get_inspector_acq_on_test_data(context,step)
    plt.plot(context.inspector.test_x, mu, linewidth=LINE_WIDTH, label='mean', linestyle='-')
    plt.plot(context.inspector.test_x, sigma, color="red", linewidth=LINE_WIDTH, label='sigma', linestyle=':')
    plt.plot(context.inspector.test_x, acq, color="orange", linewidth=LINE_WIDTH, label='acquisition func.',
             linestyle='--')
    plt.fill_between(
        context.inspector.test_x[:,0], (mu[:,0] - sigma[:,0]), (mu[:,0] + sigma[:,0]),
        alpha=TRANSPARENCY, ec='None', label='NN Uncertainty Bounds')
    # individual nets
    mu_nets = context.estimator.get_inspector_model_mus_on_test_data(context, step)
    sigma_nets = context.estimator.get_inspector_model_sigmas_on_test_data(context, step)
    acq_nets = context.estimator.get_inspector_model_acqs_on_test_data(context, step)
    colors = ['tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']
    for i in range(0, len(mu_nets)):
        plt.plot(context.inspector.test_x, mu_nets[i], color=colors[1], linewidth=LINE_WIDTH, label='mean model {}'.format(i), linestyle='-')


    plt.plot(samples_x, samples_y, 'ko', markersize=MARKER_SIZE, label='Training Data')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(-2, 2)
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.grid(True)
    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig(path)
    plt.clf()
