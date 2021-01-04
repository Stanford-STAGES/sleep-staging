import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
from h5py import File
from scipy import signal
from sklearn.preprocessing import RobustScaler, MinMaxScaler


def plot_psg_hypnogram_hypnodensity(record_id, record_predictions=None, target=None, pred=None, logits=None, interval=10, seq_idx=0, K=5, fs=128, title=None, save_path=None):
    """
    Args:
        x (ndarray): shape (N * 30 * fs, C)
        target (ndarray): shape (N, K) onehot encoded
        pred (ndarray): shape (N, K)
        logits (ndarray): shape (N * 30, K)
    Note:
        N: number of epochs (30 s)
        fs: sampling frequency
        C: number of channels
        K: number of classes
    """

    if target is None:
        target = record_predictions['true']
        # print('target.shape:', target.shape)
    if pred is None:
        pred = record_predictions['predicted']
        # print('pred.shape:', pred.shape)
    if logits is None:
        logits = record_predictions['logits']
        # print('logits.shape:', logits.shape)
    seqs = record_predictions['seq_nr']
    # print(seqs)
    # print('seqs.shape:', seqs.shape)
    ss = record_predictions['stable_sleep']
    # print('ss.shape:', ss.shape)

    try:
        with File(os.path.join('./data/ssc_wsc/raw/5min/test', record_id), 'r') as f:
            x = f['M'][:]  # (N, K, T)
    except FileNotFoundError:
        with File(os.path.join('./data/ssc_wsc/raw/5min/train', record_id), 'r') as f:
            x = f['M'][:]  # (N, K, T)

    time = (np.arange(0, np.prod(x.shape) // 5).reshape(-1, 5 * 60 * fs) / (fs * 60))[seqs]  # (M, T)
    x = x[seqs]  # (M, K, T)
    M, _, T = x.shape

    # reshape to 10 min intervals
    if M % 2 == 1:
        time = np.pad(time, [(0, 1), (0, 0)])
        x = np.pad(x, [(0, 1), (0, 0), (0, 0)])
        target = np.pad(target, [(0, 10), (0, 0)])
        pred = np.pad(pred, [(0, 10), (0, 0)])
        logits = np.pad(logits, [(0, 300), (0, 0)])
    
#     print(time.shape)
#     print('x.shape', x.shape)
    t = (time.reshape(-1)
             .reshape(-1, interval * 60 * fs))[seq_idx]
    x = (x.transpose(0, 2, 1)  # (M, T, K)
          .reshape(-1, K))  # (MxT, K)
    x = ((x / 
          x.max(axis=0))
           .reshape(-1, interval * 60 * fs, K))[seq_idx]  # (L, K)
#     print('t.shape:', t.shape)
#     print('x.shape:', x.shape)
#     print('target.shape:', target.shape)
#     print('pred.shape:', pred.shape)
#     print('logits.shape:', logits.shape)
    target = (target.transpose(1, 0)
                    .reshape(K, -1, interval * 2)
                    .transpose(1, 2, 0))[seq_idx]
    pred = (pred.transpose(1, 0)
                .reshape(K, -1, interval * 2)
                    .transpose(1, 2, 0))[seq_idx]
    logits = (logits.transpose(1, 0)
                    .reshape(K, -1, interval * 60)
                    .transpose(1, 2, 0))[seq_idx]
#     print('target.shape:', target.shape)
#     print('pred.shape:', pred.shape)
#     print('logits.shape:', logits.shape)

    # data = x.cpu().numpy()[seg_nr].T
    # data = RobustScaler(quantile_range=(0.25, 0.75)).fit_transform(data)
    if target.shape[-1] != K:
        target = target.transpose()
    hypnogram = target.argmax(axis=-1)
    # print(hypnogram.shape)
    # print(hypnogram)
    if pred.shape[-1] != K:
        pred = pred.transpose()
    if logits.shape[-1] != K:
        logits = logits.transpose()
    # if np.abs(1.0 - pred[0, :].sum()) < 1e-8:
    #     pred = softmax(pred, axis=-1)
    # if np.abs(1.0 - logits[0, :].sum()) < 1e-8:
    #     print('do softmax')
    #     logits = softmax(logits, axis=-1)
    n_epochs = len(hypnogram)
    # print('n_epochs:', n_epochs)
    
    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    
    # Setup colors
    cmap = np.array(
        [[0.4353, 0.8157, 0.9353], # W
         [0.9490, 0.9333, 0.7725], # N1
         [0.9490, 0.6078, 0.5118], # N2
         [0.6863, 0.2078, 0.2784], # N3
         [0.0000, 0.4549, 0.7373]],# R
    )
    displacement = np.vstack(
        [0 * np.ones(x.shape[0]), 1 * np.ones(x.shape[0]), 2 * np.ones(x.shape[0]), 3 * np.ones(x.shape[0]), 4 * np.ones(x.shape[0])]
    ).T
    
    # Setup figure
    fig, axes = plt.subplots(
        figsize=(20, 4), 
        nrows=3, 
        ncols=2, 
        squeeze=True, 
        dpi=150, 
        gridspec_kw={
            'height_ratios': [3, 1, 1], 
            'width_ratios': [10, 1],
            'wspace': 0.05
        }
    )
    if title is not None:
        title += f' | Seq. idx {seq_idx}'
    else:
        title = f'Seq. idx {seq_idx}'
    fig.suptitle(title)
    
    # Plot signal data
#     axes[0, 0].plot(t[::2], x[::2] / 200 - displacement[::2], "gray", linewidth=0.15)
    axes[0, 0].plot(t, x - displacement, "k", linewidth=0.15)
    axes[0, 0].set_yticks([0, -1, -2, -3, -4])
    axes[0, 0].set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    axes[0, 0].set_xlim(t[0], t[-1])
    axes[0, 0].get_xaxis().set_visible(False)
    
    # Plot power spectral data
    nperseg = x.shape[0] // 8
    nfft = int(2 ** np.ceil(np.log2(np.abs(nperseg))))
#     print('nperseg: ', nperseg)
#     print('nfft: ', nfft)
    f, Pxx = signal.welch(x[:, 0], fs=128, nperseg=nperseg, nfft=nfft, average='median')
#     Pxx_norm = Pxx / Pxx.max(axis=0)
    Pxx_norm = Pxx / Pxx.max()
#     Pxx_norm = Pxx
#     f_displacement = np.vstack(
#         [0 * np.ones(Pxx.shape[0]), 1 * np.ones(Pxx.shape[0]), 2 * np.ones(Pxx.shape[0]), 3 * np.ones(Pxx.shape[0]), 4 * np.ones(Pxx.shape[0])]
#     ).T
#     print(f)
#     print('f.shape: ', f.shape)
#     print('Pxx.shape: ', Pxx.shape)
#     print('Pxx.max(): ', Pxx.max())
    gs = axes[0, 1].get_gridspec()
    for ax in axes[:, 1]:
        ax.remove()
    bigax = fig.add_subplot(gs[:, 1])
    bigax.plot(f, Pxx_norm, 'k', linewidth=0.2)
#     axes[0, 1].semilogx(f, Pxx_norm - f_displacement, 'k', linewidth=0.2)
#     axes[0, 1].get_xaxis().set_visible(False)
    bigax.set_xlabel('Frequency (Hz)')
    bigax.set_xticks([0, 10, 20, 30])
    bigax.set_xlim(0, 35)
    bigax.set_ylim(0, 0.5)
    bigax.get_yaxis().set_visible(False)
    
    # Add vertical divider lines
    vline_coords = np.arange(0, len(hypnogram)) / 2 + t[0]
    for xc, h in zip(vline_coords, hypnogram):
        axes[0, 0].axvline(xc + 0.5, linewidth=0.5, color="grey")
    
    # Plot 1 s hypnodensity
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(axes[1, 0].fill_between(np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :], linewidth=0.0, step='post'))
    axes[1, 0].get_xaxis().set_visible(False)
    axes[1, 0].set_xlim([0, len(hypnodensity.T) - 1])
    axes[1, 0].set_ylim([0.0, 1.0])
    axes[1, 0].set_ylabel('1 s')
    plt.setp(axes[1, 0].get_yticklabels(), visible=False)
    axes[1, 0].tick_params(axis='both', which='both', length=0)
    
    # Add vertical divider lines
    vline_coords = np.arange(0, hypnodensity.shape[1] // 30)
    for xc, h in zip(vline_coords, hypnogram):
        axes[1, 0].axvline(xc*30, linewidth=0.5, color="grey")

    # Plot predicted hypnodensity at 30 s
    h = pred.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(axes[2, 0].fill_between(np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :], linewidth=0.0, step='post'))
#     axes[2, 0].get_xaxis().set_visible(False)
    plt.setp(axes[2, 0].get_yticklabels(), visible=False)
    axes[2, 0].set_xlabel('Time (min)')
    axes[2, 0].set_ylabel('30 s')
    axes[2, 0].set_xlim(0, hypnodensity.shape[1] - 1)
    axes[2, 0].set_ylim(0.0, 1.0)
    axes[2, 0].tick_params(axis='y', which='both', length=0)
    axes[2, 0].set_xticks(np.arange(0, hypnodensity.shape[1] + 1, 2))
    axes[2, 0].set_xticklabels(np.arange(0, hypnodensity.shape[1] + 1, 2) // 2 )
    
    # Add vertical divider lines, manual and predicted 30 s hypnograms
    for xc, h_pred, h_true in zip(vline_coords, hypnodensity.argmax(0)[:-1], hypnogram):
        axes[2, 0].axvline(xc + 1, linewidth=0.5, color="grey") # Divider line
        txt = axes[2, 0].text(xc + 0.5, 7-1.25, hypnogram_dict[h_true], horizontalalignment="center", color=cmap[h_true])  # manual hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='grey')])
        txt = axes[2, 0].text(xc + 0.5, 1.075, hypnogram_dict[h_pred], horizontalalignment="center", color=cmap[h_pred])# automatic hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='gray')])
    
#     # Add manual and predicted 30 s hypnograms
#     for xc, h in zip(vline_coords, hypnodensity.argmax(0)[:-1]):
    
    # Add predicted 30 s hypnogram at the bottom
#     vline_coords = np.arange(0, hypnodensity.shape[1] - 1)
#     print(vline_coords)
#     for xc, h in zip(vline_coords, hypnodensity.argmax(0)[:-1]):
        
    # Add text objects
    axes[2, 0].text(-0.2, 7-1.25, 'Manual', ha='right', color='grey')
    axes[2, 0].text(-0.2, 1.075, 'Automatic', ha='right', color='grey')
    
    # ADDITIONAL HOUSEKEEPING
#     fig.delaxes(axes[1, 1])
#     fig.delaxes(axes[2, 1])
#     axes[1, 1].get_xaxis().set_visible(False)
#     axes[1, 1].get_yaxis().set_visible(False)
#     axes[2, 1].get_xaxis().set_visible(False)
#     axes[2, 1].get_yaxis().set_visible(False)
#     fig.tight_layout()
        
    # Save figure
    if save_path is not None:
        fig.savefig(f'results/{save_path}', dpi=150, bbox_inches='tight', pad_inches=0)
        
#     plt.show()
    plt.close()
    


def plot_hypnodensity(logits, preds, trues, title=None, save_path=None):
    
    # Setup title
    f, ax = plt.subplots(nrows=4, figsize=(20, 5), dpi=400)
    f.suptitle(title)
    
    # Setup colors
    cmap = np.array(
        [[0.4353, 0.8157, 0.9353], # W
         [0.9490, 0.9333, 0.7725], # N1
         [0.9490, 0.6078, 0.5118], # N2
         [0.6863, 0.2078, 0.2784], # N3
         [0.0000, 0.4549, 0.7373]],# R
    )
    
    # Plot the hypnodensity
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(ax[0].fill_between(np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :], linewidth=0.0, step='post'))
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel('1 s')
    plt.setp(ax[0].get_yticklabels(), visible=False)
    ax[0].tick_params(axis='both', which='both', length=0)
    
    # Create legend
    legend_elements = [mpl.patches.Patch(facecolor=cm, edgecolor=cm, label=lbl) for cm, lbl in zip(cmap, ['W', 'N1', 'N2', 'N3', 'REM'])]
    ax[0].legend(handles=legend_elements, loc='lower center', bbox_to_anchor=[0.5, 1.0], ncol=5)
#     sns.despine(top=True, bottom=True, left=True, right=True)
#     plt.tight_layout()

    # Plot predicted hypnodensity at 30 s
    h = preds.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(ax[1].fill_between(np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :], linewidth=0.0, step='post'))
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_ylabel('30 s')
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[1].tick_params(axis='both', which='both', length=0)

    # Plot predicted hyponogram
    ax[2].plot(preds.argmax(axis=-1))
    ax[2].set_xlim(0, trues.shape[0] - 1)
    ax[2].set_ylim(-.5, 4.5)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_yticks([0, 1, 2, 3, 4])
    ax[2].set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
    ax[2].set_ylabel('Automatic')

    # Plot true hyponogram
    ax[3].plot(trues.argmax(axis=-1))
    ax[3].set_xlim(0, trues.shape[0] - 1)
    ax[3].set_ylim(-0.5, 4.5)
    ax[3].set_yticks([0, 1, 2, 3, 4])
    ax[3].set_yticklabels(['W', 'N1', 'N2', 'N3', 'R'])
    ax[3].set_xticks(np.arange(0, trues.shape[0] - 1, 20))
    ax[3].set_xticklabels(np.arange(0, trues.shape[0] - 1, 20) * 30 // 60)
    ax[3].set_xlabel('Time (min)')
    ax[3].set_ylabel('Manual')
    
    # Save figure
    if save_path is not None:
        f.savefig(f'results/{save_path}', dpi=300, bbox_inches='tight', pad_inches=0)
#     plt.close()
    plt.show()


# def plot_psg_hypnogram_hypnodensity(x, target, pred):
#     """[summary]

#     Args:
#         x ([type]): [description]
#         target ([type]): [description]
#         pred ([type]): [description]
#     """


def plot_segment(x, target, pred):

    seg_nr = 6
    data = x.cpu().numpy()[seg_nr].T
    # data = RobustScaler(quantile_range=(0.25, 0.75)).fit_transform(data)
    hypnogram = target.cpu().numpy().argmax(axis=1)[seg_nr, ::30]
    hypnodensity = pred.cpu().softmax(dim=1).numpy()[seg_nr]
    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    displacement = np.vstack(
        [
            0 * np.ones(len(data)),
            1 * np.ones(len(data)),
            2 * np.ones(len(data)),
            3 * np.ones(len(data)),
            4 * np.ones(len(data)),
        ]
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
        [
            [0.6863, 0.2078, 0.2784],
            [0.9490, 0.6078, 0.5118],
            [0.9490, 0.9333, 0.7725],
            [0.4353, 0.8157, 0.9353],
            [0.0000, 0.4549, 0.7373],
        ]
    )
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            plt.fill_between(
                np.arange(hypnodensity.shape[1]), y_[n, :], y_[n + 1, :], edgecolor="face", facecolor=cmap[n, :]
            )
        )
    axes[1].set_ylim([0.0, 1.0])
    axes[1].set_yticks([])
    plt.subplots_adjust(hspace=0)
    # plt.tight_layout(h_pad=-0.5)
    # plt.savefig("prediction.png", dpi=600)  # , bbox_inches='tight')

    return fig
