import os
import pickle

import matplotlib.pyplot as plt

MARKER_SIZE = 6
LINE_WIDTH = 2
TRANSPARENCY = 0.2


def plot_prediction(path_to_file, title):
    c_r = 1
    with open(path_to_file, 'rb') as handle:
        data = pickle.load(handle)


    fig = plt.figure(figsize=(16, 9))
    plt.plot(data["sample_x"], data["sample_y"], 'ko', markersize=MARKER_SIZE, label='Training Data')
    # if plotaugmented:
    #    plt.plot(x_aug[:, :-1], y_aug, 'ko', label='Artificial Data', markersize=markersize, color='red')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(data["test_x"], data["test_y"], label='True Function', color='black')
    plot_nnub = plt.plot(data["test_x"], data["test_means"], linewidth=LINE_WIDTH, label='mean', linestyle='-')
    plt.plot(data["test_x"], c_r*data["test_sigmas"], color="red", linewidth=LINE_WIDTH, label='{}*r'.format(c_r), linestyle=':')
    plt.plot(data["test_x"], data["test_acqs"], color="orange", linewidth=LINE_WIDTH, label='acquisition func.', linestyle='--')
    plt.plot(data["next_samples_x"], data["next_samples_y"],'ko', color="red", label='to evaluate')
    plt.fill_between(
        data["test_x"][:,0],
        (data["test_means"][:,0]-(c_r*data["test_sigmas"][:,0])),
        (data["test_means"][:,0]+(c_r*data["test_sigmas"][:,0])),
        alpha=TRANSPARENCY, ec='None', label='NN Uncertainty Bounds. {}*r'.format(c_r))
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.grid(True)
    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig('{}.png'.format(path_to_file.split(".pickle")[0]))
    plt.clf()


def plot_regret(path_to_dirs, title):
    all_data = []
    for path in path_to_dirs:
        data_per_model = []
        for file in os.listdir(path):
            if file.endswith(".pickle"):
                with open(os.path.join(path, file), 'rb') as handle:
                    data_per_model.append(pickle.load(handle))
        all_data.append(data_per_model)

    fig = plt.figure(figsize=(16, 9))

    for i, data in enumerate(all_data):
        if len(data) > 0:
            plt.plot(list(map(lambda x: x["step"], data)), list(map(lambda x: x["relative_regret_y"], data)), markersize=MARKER_SIZE, label=data[0]["model"])
            # plt.plot(list(map(lambda x: x["step"], data)), list(map(lambda x: abs(np.max(x["sample_y"][:,0])-(-0.39788735773)), data)), markersize=MARKER_SIZE, label=path_to_dirs[i].rsplit("/",1)[-1])
    plt.xlabel("steps")
    plt.ylabel("relative regret (pre)")
    plt.legend(loc='best', shadow=True, fontsize='small')
    plt.grid(True)
    plt.title(title, fontsize='small')
    plt.tight_layout()
    fig.savefig('{}/{}.png'.format(path_to_dirs[0].rsplit("/",1)[0], title))
    plt.clf()


