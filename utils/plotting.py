import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def plot_segment(x, target, pred):

    seg_nr = 6
    data = x.cpu().numpy()[seg_nr].T
    # data = RobustScaler(quantile_range=(0.25, 0.75)).fit_transform(data)
    hypnogram = target.cpu().numpy().argmax(axis=1)[seg_nr, ::30]
    hypnodensity = pred.cpu().softmax(dim=1).numpy()[seg_nr]
    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    displacement = np.vstack(
        [0 * np.ones(len(data)), 1 * np.ones(len(data)), 2 * np.ones(len(data)), 3 * np.ones(len(data)), 4 * np.ones(len(data))]
    ).T

    fig, axes = plt.subplots(figsize=(7, 4), nrows=2, sharex=True, squeeze=True, dpi=600)
    axes[0].plot(data / 200 - displacement, "-k", linewidth=0.5)
    axes[0].set_xlim([0, len(data)])
    axes[0].set_ylim([-4.5, 0.5])
    axes[0].set_xticks([])
    axes[0].set_yticks([0, -1, -2, -3, -4])
    axes[0].set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    vline_coords = np.arange(30 * 128, len(data), 30 * 128)
    for xc, h in zip(vline_coords, hypnogram):
        axes[0].axvline(xc, linewidth=0.5, color="grey")
        axes[0].text(xc - 15 * 128, 0.75, hypnogram_dict[h], horizontalalignment="center")
    axes[0].text(xc + 15 * 128, 0.75, hypnogram_dict[hypnogram[-1]], horizontalalignment="center")
    cmap = np.array(
        [[0.6863, 0.2078, 0.2784], [0.9490, 0.6078, 0.5118], [0.9490, 0.9333, 0.7725], [0.4353, 0.8157, 0.9353], [0.0000, 0.4549, 0.7373]]
    )
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(plt.fill_between(np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :]))
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_yticks([])
    plt.subplots_adjust(hspace=0)
    # plt.tight_layout(h_pad=-0.5)
    # plt.savefig("prediction.png", dpi=600)  # , bbox_inches='tight')

    return fig
