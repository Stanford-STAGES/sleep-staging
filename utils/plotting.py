import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np
import plotly.express as px
from h5py import File
from scipy import signal
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from .multitaper_spectrogram import multitaper_spectrogram, nanpow2db


def view_record(
    record_id,
    record_predictions=None,
    target=None,
    pred=None,
    logits=None,
    fs=128,
    spectrum_type="multitaper",
    spectral_kws={'window_dur': 30, 'window_step': 5, 'delta_f': 1},
    title=None,
    save_path=None,
    verbose=True,
):
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
        target = record_predictions["true"]
    if pred is None:
        pred = record_predictions["predicted"]
    if logits is None:
        logits = record_predictions["logits"]
    seqs = record_predictions["seq_nr"]
#     print('target.shape:', target.shape)
#     print('pred.shape:', pred.shape)
#     print('logits.shape:', logits.shape)

#     print(f"Current file: {record_id}")
    epoch_nrs = seqs

    try:
        with File(os.path.join("./data/test/raw", record_id), "r") as f:
            x = f["M"][seqs]  # (N, K, T)
    except FileNotFoundError:
        with File(os.path.join("./data/train/raw", record_id), "r") as f:
            x = f["M"][seqs]  # (N, K, T)

    M, K, T = x.shape
    t = np.arange(0, M * T)
    x = x.transpose(1, 0, 2).reshape(K, -1).T
#     print('x.shape:', x.shape)
#     print('t.shape:', t.shape)
#     x = RobustScaler().fit_transform(x)
#     print('target.shape:', target.shape)

    if target.shape[-1] != K:
        target = target.transpose()
    hypnogram = target.argmax(axis=-1)
    if pred.shape[-1] != K:
        pred = pred.transpose()
    if logits.shape[-1] != K:
        logits = logits.transpose()
    n_epochs = len(hypnogram)

    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    # Setup colors
    cmap = np.array(
        [
            [0.4353, 0.8157, 0.9353],  # W
            [0.9490, 0.9333, 0.7725],  # N1
            [0.9490, 0.6078, 0.5118],  # N2
            [0.6863, 0.2078, 0.2784],  # N3
            [0.0000, 0.4549, 0.7373],  # R
        ],
    )
    displacement = np.vstack([i * np.ones(x.shape[0]) for i in range(x.shape[1])]).T
#     print(displacement)

    # Setup figure
    fig, axes = plt.subplots(
        figsize=(18, 8),
        nrows=8,
        ncols=1,
        squeeze=True,
        dpi=150,
        gridspec_kw={
#             "height_ratios": [4, 4, 3, 3, 3, 2.5, 2.5],
            "height_ratios": [2, 2, 2, 2, 2, 0.5, 1, 1],
            "hspace": 0.05,
        },
    )
    if title is None:
        title = f"{record_id}"
    fig.suptitle(title)

    # Plot power spectral data
#     multitaper_spectrogram(x[:, 0], fs, [0, 25], 15., 29, [30, 5])
#     return
    current_ax = axes[0]
    if spectrum_type == "multitaper":
        if spectral_kws is None:
            window_dur=3
            window_step=0.1
            delta_f=1.5
        mts_params = dict(
            frequency_range=[0, 25],
            time_bandwidth=spectral_kws['window_dur'] * spectral_kws['delta_f'] / 2,
            window_params=[spectral_kws['window_dur'], spectral_kws['window_step']],
#             min_nfft=int(2 ** np.ceil(np.log2(np.abs(spectral_kws['window_dur'] * fs)))),
            detrend_opt='linear'
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 0], fs, **mts_params, plot_on=False, verbose=verbose)
#         spec_f, spec_t, Zxx = signal.stft(x[:, 1], fs, nperseg=30 * 128, noverlap=25*128, nfft=int(2 ** np.ceil(np.log2(30 * fs))))
#         Zxx = Zxx.T
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(x[:, 1], signal.morlet2, width, w=w).T
    vmin, vmax = np.quantile(2 * nanpow2db(np.abs(Zxx)).T, [0.1, 0.9])
    im = current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), 2 * nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto", vmin=vmin, vmax=vmax
    )
#     plt.colorbar(im, orientation='horizontal', anchor=(1., 1.))
    current_ax.get_xaxis().set_visible(False)

    # Create legend
    legend_elements = [
        mpl.patches.Patch(facecolor=cm, edgecolor=cm, label=lbl)
        for cm, lbl in zip(cmap, ["W", "N1", "N2", "N3", "REM"])
    ]
    current_ax.legend(handles=legend_elements, loc="lower right", bbox_to_anchor=[1.0, 1.0], ncol=5)

    current_ax = axes[1]
    if spectrum_type == "multitaper":
        if spectral_kws is None:
            window_dur=3
            window_step=0.1
            delta_f=1.5
        mts_params = dict(
            frequency_range=[0, 25],
            time_bandwidth=spectral_kws['window_dur'] * spectral_kws['delta_f'] / 2,
            window_params=[spectral_kws['window_dur'], spectral_kws['window_step']],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(spectral_kws['window_dur'] * fs)))),
            detrend_opt='linear'
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 1], fs, **mts_params, plot_on=False, verbose=verbose)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(x[:, 1], signal.morlet2, width, w=w).T
    vmin, vmax = np.quantile(2 * nanpow2db(np.abs(Zxx)).T, [0.1, 0.9])
    current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), 2 * nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto", vmin=vmin, vmax=vmax
    )
    current_ax.get_xaxis().set_visible(False)
#     return

    # Plot signal data
    current_ax = axes[2]
    _x = RobustScaler().fit_transform(x)
    current_ax.plot(t, _x / _x.max() - displacement, "k", linewidth=0.05)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    current_ax.set_xlim(t[0], t[-1])
    current_ax.get_xaxis().set_visible(False)

    # Add vertical divider lines
#     current_ax = axes[1]
#     vline_coords = np.arange(0, len(hypnogram)) / 2 + t[0]
#     for xc, h in zip(vline_coords, hypnogram):
#         current_ax.axvline(xc + 0.5, linewidth=0.5, color="grey")

    # Plot 1 s hypnodensity
    current_ax = axes[3]
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_xlim([0, len(hypnodensity.T) - 1])
    current_ax.set_ylim([0.0, 1.0])
    current_ax.set_ylabel("1 s")
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.tick_params(axis="both", which="both", length=0)

    # Add vertical divider lines
#     vline_coords = np.arange(0, hypnodensity.shape[1] // 30)
#     for xc, h in zip(vline_coords, hypnogram):
#         current_ax.axvline(xc * 30, linewidth=0.5, color="grey")

    # Plot predicted hypnodensity at 30 s
    current_ax = axes[4]
    h = pred.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.set_xlabel("Time (s)")
    current_ax.set_ylabel("30 s")
    current_ax.set_xlim(0, hypnodensity.shape[1] - 1)
    current_ax.set_ylim(0.0, 1.0)
    current_ax.tick_params(axis="y", which="both", length=0)
    current_ax.set_xticks(np.arange(0, len(hypnogram) + 1), 0.5)
    current_ax.set_xticklabels(np.arange(0, len(hypnogram) * 2, 1) * 15)
    current_ax.get_xaxis().set_visible(False)

    current_ax = axes[5]
#     current_ax.plot(pred.max(axis=1))
    entr = - (pred * np.log(pred)).sum(axis=-1)
    current_ax.plot(entr)
    current_ax.set_xlim(0, len(pred))
#     current_ax.get_yaxis().set_visible(False)
    current_ax.set_yticks([])
    current_ax.set_ylabel('Entr', rotation=0)
    current_ax.get_xaxis().set_visible(False)

    hypno = -(pred[:, [0, 4, 1, 2, 3]]).argmax(axis=-1)
    current_ax = axes[6]
    current_ax.plot(hypno)
    current_ax.set_xlim(0, len(hypno))
    current_ax.set_ylim(-4.5, 0.5)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["W", "R", "N1", "N2", "N3"])
    current_ax.get_xaxis().set_visible(False)

    current_ax = axes[7]
    current_ax.plot(-target[:, [0, 4, 1, 2, 3]].argmax(axis=-1))
    current_ax.set_xlim(0, len(hypnogram))
    current_ax.set_ylim(-4.5, 0.5)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["W", "R", "N1", "N2", "N3"])
    current_ax.get_xaxis().set_visible(False)

    # Add vertical divider lines, manual and predicted 30 s hypnograms
#     manual_str_placement = 8.8
#     auto_str_placement = 8.45
#     for xc, h_pred, h_true, epch_nr in zip(vline_coords, hypnodensity.argmax(0)[:-1], hypnogram, epoch_nrs):
#         current_ax.axvline(xc + 1, linewidth=0.5, color="grey")  # Divider line
#         txt = current_ax.text(
#             xc + 0.5, manual_str_placement, hypnogram_dict[h_true], horizontalalignment="center", color=cmap[h_true]
#         )  # manual hypnogram
#         txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="grey")])
#         txt = current_ax.text(
#             xc + 0.5, auto_str_placement, hypnogram_dict[h_pred], horizontalalignment="center", color=cmap[h_pred]
#         )  # automatic hypnogram
#         txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="gray")])
#     #         txt = current_ax.text(xc + 0.5, 2.4, epch_nr, horizontalalignment="center")

#     # Add text objects
#     current_ax.text(0.0, manual_str_placement, "Manual", ha="left", color="grey")
#     current_ax.text(0.0, auto_str_placement, "Automatic", ha="left", color="grey")
    #     current_ax.text(-0.025, 1.075, 'Automatic', ha='right', color='grey')

    # Save figure
    if save_path is not None:
        fig.savefig(f"results/{save_path}", dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

def extended_epoch_view(
    record_id,
    record_predictions=None,
    target=None,
    pred=None,
    logits=None,
    interval=10,
    seq_nr=0,
    fs=128,
    spectrum_type="multitaper",
    title=None,
    save_path=None,
    verbose=False,
):
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

    if record_id.split('.')[1] == 'pkl':
        record_id = record_id[6:].split('.')[0] + '.h5'

    if target is None:
        _target = record_predictions["true"]
        target = _target.copy()
    if pred is None:
        pred = record_predictions["predicted"]
    if logits is None:
        logits = record_predictions["logits"]
    yhat_3s = record_predictions['yhat_3s']
    seqs = record_predictions["seq_nr"]

    m_factor = int(5 / interval)
    seqs = np.array([s * m_factor + step for s in seqs for step in range(m_factor)])

    if verbose:
        print(f"Current file: {record_id}")
        print(f"Selected epoch: {seq_nr}")
        print(f"Min. avail. epoch: {seqs.min()}")
        print(f"Max. avail. epoch: {seqs.max()}")
    assert (seq_nr <= seqs.max()) & (seq_nr >= seqs.min()), (
        f"Sequence nr. must be between {seqs.min()} and {seqs.max()}. " f"Supplied index {seq_nr}."
    )
    epoch_nrs = seqs

    try:
        with File(os.path.join("./data/test/raw", record_id), "r") as f:
            x = f["M"][:]  # (N, K, T)
    except FileNotFoundError:
        with File(os.path.join("./data/train/raw", record_id), "r") as f:
            x = f["M"][:]  # (N, K, T)
    #     try:
    #         with File(os.path.join("./data/ssc_wsc/raw/5min/test", record_id), "r") as f:
    #             x = f["M"][:]  # (N, K, T)
    #     except FileNotFoundError:
    #         with File(os.path.join("./data/ssc_wsc/raw/5min/train", record_id), "r") as f:
    #             x = f["M"][:]  # (N, K, T)
    N, K, T = x.shape
    time = np.arange(0, np.prod(x.shape) // K).reshape(N, T)

    if (N * 5 * 60) % (60 * interval):
        pad_amount = int((N * 5 * 60) % (60 * interval))  # pad amount in seconds
        print(f"Padding: {pad_amount} seconds")
        time = np.pad(time, [(0, np.ceil(pad_amount / (5 * 60)).astype(int)), (0, 0)])
        x = np.pad(x, [(0, np.ceil(pad_amount / (5 * 60)).astype(int)), (0, 0), (0, 0)])
        target = np.pad(target, [(0, np.ceil(pad_amount / (0.5 * 60)).astype(int)), (0, 0)])
        pred = np.pad(pred, [(0, np.ceil(pad_amount / (0.5 * 60)).astype(int)), (0, 0)])
        logits = np.pad(logits, [(0, pad_amount), (0, 0)])

    x = (x.transpose(1, 0, 2).reshape(K, -1).reshape(K, -1, int(interval * 60 * fs)).transpose(1, 0, 2))[
        seqs
    ]  # (M, K, T)
    time = (time.reshape(-1).reshape(-1, int(interval * 60 * fs)) / (fs * 60))[seqs]  # (M, T)
    M, _, T = x.shape
    seq_idx = int(np.argwhere(seqs == seq_nr).flatten())
    seq_idx_vec = slice(seq_idx - 1, seq_idx + 2)
    t = time[seq_idx_vec].flatten()
    x = x.transpose(0, 2, 1).reshape(-1, K)  # (M, T, K)  # (MxT, K)
    # x = ((x / x.max(axis=0)).reshape(-1, int(interval * 60 * fs), K))[seq_idx_vec]  # (L, K)
    x = (x.reshape(-1, int(interval * 60 * fs), K))[seq_idx_vec]  # (L, K)
    x = x.transpose(2, 0, 1).reshape(5, -1).T

    target = (target.transpose(1, 0).reshape(K, -1, int(2 * interval)).transpose(1, 2, 0))[
        seq_idx_vec
    ]  # (interval, 1, K)
    target = np.moveaxis(target, -1, 0).reshape(K, -1).T
    pred = (pred.transpose(1, 0).reshape(K, -1, int(2 * interval)).transpose(1, 2, 0))[seq_idx_vec].reshape(-1, K)
    logits = (logits.transpose(1, 0).reshape(K, -1, int(interval * 60)).transpose(1, 2, 0))[seq_idx_vec]  # (M, T, K)
    logits = np.moveaxis(logits, -1, 0).reshape(K, -1).T
    yhat_3s = (yhat_3s.transpose(1, 0).reshape(K, -1, int(interval * 60 / 3)).transpose(1, 2, 0))[seq_idx_vec]
    yhat_3s = np.moveaxis(yhat_3s, -1, 0).reshape(K, -1).T
    epoch_nrs = epoch_nrs.reshape(-1, int(interval * 2))[seq_idx_vec]

    if target.shape[-1] != K:
        target = target.transpose()
    hypnogram = target.argmax(axis=-1)
    if pred.shape[-1] != K:
        pred = pred.transpose()
    if logits.shape[-1] != K:
        logits = logits.transpose()
    if yhat_3s.shape[-1] != K:
        yhat_3s = yhat_3s.tranpose()
    n_epochs = len(hypnogram)

    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    # Setup colors
    cmap = np.array(
        [
            [0.4353, 0.8157, 0.9353],  # W
            [0.9490, 0.9333, 0.7725],  # N1
            [0.9490, 0.6078, 0.5118],  # N2
            [0.6863, 0.2078, 0.2784],  # N3
            [0.0000, 0.4549, 0.7373],
        ],  # R
    )
    displacement = np.vstack(
        [
            0 * np.ones(x.shape[0]),
            1 * np.ones(x.shape[0]),
            2 * np.ones(x.shape[0]),
            3 * np.ones(x.shape[0]),
            4 * np.ones(x.shape[0]),
        ]
    ).T

    # Setup figure
    fig, axes = plt.subplots(
        figsize=(20, 8),
        nrows=6,
        ncols=1,
        squeeze=True,
        dpi=150,
        gridspec_kw={"height_ratios": [3, 3, 3, 1, 1, 1], "hspace": 0.05,},
    )
    if title is None:
        title = f"{record_id} | Seq. nr. {seq_nr}"
    fig.suptitle(title)

    # Plot power spectral data
    current_ax = axes[0]
    if spectrum_type == "multitaper":
        window_dur = 3
        window_step = 0.05
        delta_f = 2
        mts_params = dict(
            frequency_range=[0, 25],
            time_bandwidth=window_dur * delta_f / 2,
            window_params=[window_dur, window_step],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs)))),
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 0], fs, **mts_params, plot_on=False, verbose=verbose)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(x[:, 0], signal.morlet2, width, w=w).T
    vmin, vmax = np.quantile(2 * nanpow2db(np.abs(Zxx)).T, [0.1, 0.9])
    im = current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), 2 * nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto", vmin=vmin, vmax=vmax
    )
    current_ax.set_ylabel('EEG C')
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_yticks([0, 5, 10, 15, 20])
    current_ax.set_yticklabels([0, 5, 10, 15, 20])
    # current_ax.set_xticklabels(np.arange(0, len(hypnogram) * 2, 1) * 15)

    # Create legend
    legend_elements = [
        mpl.patches.Patch(facecolor=cm, edgecolor=cm, label=lbl)
        for cm, lbl in zip(cmap, ["W", "N1", "N2", "N3", "REM"])
    ]
    # current_ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(1.00, 0.), ncol=1, labelspacing=1.5)
    current_ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 1.6), ncol=5)

    # Add a colorbar
    cbaxes = fig.add_axes([0.125, 0.96, 0.15, 0.02])
    cb = plt.colorbar(im, orientation='horizontal', cax=cbaxes, format='%d dB')

    current_ax = axes[1]
    if spectrum_type == "multitaper":
        window_dur = 3
        window_step = 0.05
        delta_f = 2
        mts_params = dict(
            frequency_range=[0, 25],
            time_bandwidth=window_dur * delta_f / 2,
            window_params=[window_dur, window_step],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs)))),
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 1], fs, **mts_params, plot_on=False, verbose=verbose)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(x[:, 0], signal.morlet2, width, w=w).T
    vmin, vmax = np.quantile(2 * nanpow2db(np.abs(Zxx)).T, [0.1, 0.9])
    current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), 2 * nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto",  vmin=vmin, vmax=vmax
    )
    current_ax.set_ylabel('EEG O')
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_yticks([0, 5, 10, 15, 20])
    current_ax.set_yticklabels([0, 5, 10, 15, 20])

    # Plot signal data
    current_ax = axes[2]
    current_ax.plot(x / (x.max(axis=0) + 1e-8) - displacement, "k", linewidth=0.25)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    # current_ax.set_xlim(t[0], t[-1])
    current_ax.set_xlim(0, len(x) - 1)
    current_ax.get_xaxis().set_visible(False)

    # Add vertical divider lines
    current_ax = axes[2]
    # vline_coords = np.arange(0, len(hypnogram)) / 2 + t[0]
    vline_coords = np.arange(0, len(x), 30 * 128)
    for xc in vline_coords:
        current_ax.axvline(xc, linewidth=0.5, color="grey")

    # Plot 1 s hypnodensity
    current_ax = axes[3]
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_xlim([0, len(hypnodensity.T) - 1])
    current_ax.set_ylim([0.0, 1.0])
    current_ax.set_ylabel("1 s")
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.tick_params(axis="both", which="both", length=0)

    # Add vertical divider lines
    vline_coords = np.arange(0, hypnodensity.shape[1] // 30)
    for xc, h in zip(vline_coords, hypnogram):
        current_ax.axvline(xc * 30, linewidth=0.5, color="grey")

    # Plot 3 s hypnodensity
    current_ax = axes[4]
    h = yhat_3s.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_xlim([0, len(hypnodensity.T) - 1])
    current_ax.set_ylim([0.0, 1.0])
    current_ax.set_ylabel("3 s")
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.tick_params(axis="both", which="both", length=0)

    # Add vertical divider lines
    vline_coords = np.arange(0, hypnodensity.shape[1] // 10)
    for xc, h in zip(vline_coords, hypnogram):
        current_ax.axvline(xc * 10, linewidth=0.5, color="grey")

    # Plot predicted hypnodensity at 30 s
    current_ax = axes[5]
    h = pred.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.set_xlabel("Time (s)")
    current_ax.set_ylabel("30 s")
    current_ax.set_xlim(0, hypnodensity.shape[1] - 1)
    current_ax.set_ylim(0.0, 1.0)
    current_ax.tick_params(axis="y", which="both", length=0)
    current_ax.set_xticks(np.arange(0, len(hypnogram) + 1), 0.5)
    current_ax.set_xticklabels(np.arange(0, len(hypnogram) * 2, 1) * 15)

    # Add vertical divider lines, manual and predicted 30 s hypnograms
    manual_str_placement = 13.
    auto_str_placement = 12.65
    for xc, h_pred, h_true, epch_nr in zip(vline_coords, hypnodensity.argmax(0)[:-1], hypnogram, epoch_nrs):
        current_ax.axvline(xc + 1, linewidth=0.5, color="grey")  # Divider line
        txt = current_ax.text(
            xc + 0.5, manual_str_placement, hypnogram_dict[h_true], horizontalalignment="center", color=cmap[h_true]
        )  # manual hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="grey")])
        txt = current_ax.text(
            xc + 0.5, auto_str_placement, hypnogram_dict[h_pred], horizontalalignment="center", color=cmap[h_pred]
        )  # automatic hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="gray")])
    #         txt = current_ax.text(xc + 0.5, 2.4, epch_nr, horizontalalignment="center")

    # Add text objects
    current_ax.text(0.0, manual_str_placement, "Manual", ha="left", color="grey")
    current_ax.text(0.0, auto_str_placement, "Automatic", ha="left", color="grey")
    #     current_ax.text(-0.025, 1.075, 'Automatic', ha='right', color='grey')

    # Add manual hypnogram for whole night
    current_ax = fig.add_axes([0.125, -0.05, 0.775, 0.1])
    current_ax.plot(seqs, -_target[:, [0, 4, 1, 2, 3]].argmax(axis=-1), color='k', drawstyle='steps-post')
    current_ax.set_xlim(0, len(_target))
    current_ax.set_ylim(-4.5, 0.5)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["W", "R", "N1", "N2", "N3"])
    current_ax.get_xaxis().set_visible(False)
    current_ax.axvspan(seqs[seq_idx_vec][0], seqs[seq_idx_vec][-1] + 1, facecolor='r', alpha=0.5)

    # Save figure
    if save_path is not None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)

    plt.close()


def plot_psg_hypnogram_hypnodensity(
    record_id,
    record_predictions=None,
    target=None,
    pred=None,
    logits=None,
    interval=10,
    seq_nr=0,
    fs=128,
    spectrum_type="multitaper",
    title=None,
    save_path=None,
    verbose=False,
):
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
        target = record_predictions["true"]
        # print('target.shape:', target.shape)
    if pred is None:
        pred = record_predictions["predicted"]
        # print('pred.shape:', pred.shape)
    if logits is None:
        logits = record_predictions["logits"]
        # print('logits.shape:', logits.shape)
    seqs = record_predictions["seq_nr"]

    #     if interval != 5:
    m_factor = int(5 / interval)
    seqs = np.array([s * m_factor + step for s in seqs for step in range(m_factor)])
    #     print('seq:', seqs)
    #         print('seq_new:', [s * m_factor + step for s in seqs for step in range(m_factor)])

    print(f"Current file: {record_id}")
    print(f"Min. avail. epoch: {seqs.min()}")
    print(f"Max. avail. epoch: {seqs.max()}")
    assert (seq_nr <= seqs.max()) & (seq_nr >= seqs.min()), (
        f"Sequence nr. must be between {seqs.min()} and {seqs.max()}. " f"Supplied index {seq_nr}."
    )
    #     epoch_nrs = seqs
    epoch_nrs = np.array([x * m_factor + y for x in seqs for y in range(int(10 / m_factor))])
    print(epoch_nrs.shape)
    # print(seqs)
    # print('seqs.shape:', seqs.shape)
    ss = record_predictions["stable_sleep"]
    # print('ss.shape:', ss.shape)

    try:
        with File(os.path.join("./data/ssc_wsc/raw/5min/test", record_id), "r") as f:
            x = f["M"][:]  # (N, K, T)
    except FileNotFoundError:
        with File(os.path.join("./data/ssc_wsc/raw/5min/train", record_id), "r") as f:
            x = f["M"][:]  # (N, K, T)
    N, K, T = x.shape
    time = np.arange(0, np.prod(x.shape) // K).reshape(N, T)
    print("time.shape: ", time.shape)

    # Test the interval is conforming to the number of 5 min sequences in file
    #     print(f'N % 2 == {N % 2}')
    #     if (N % 2 == 1):
    if (N * 5 * 60) % (60 * interval):
        pad_amount = int((N * 5 * 60) % (60 * interval))  # pad amount in seconds
        print(f"Padding: {pad_amount} seconds")
        #         print('time.shape: ', time.shape)
        time = np.pad(time, [(0, np.ceil(pad_amount / (5 * 60)).astype(int)), (0, 0)])
        #         print('time.shape: ', time.shape)
        #         print('x.shape:', x.shape)
        x = np.pad(x, [(0, np.ceil(pad_amount / (5 * 60)).astype(int)), (0, 0), (0, 0)])
        #         print('x.shape:', x.shape)
        #         print('target.shape:', target.shape)
        target = np.pad(target, [(0, np.ceil(pad_amount / (0.5 * 60)).astype(int)), (0, 0)])
        #         print('target.shape:', target.shape)
        #         print('pred.shape:', pred.shape)
        pred = np.pad(pred, [(0, np.ceil(pad_amount / (0.5 * 60)).astype(int)), (0, 0)])
        #         print('pred.shape:', pred.shape)
        #         print('logits.shape:', logits.shape)
        logits = np.pad(logits, [(0, pad_amount), (0, 0)])
    #         print('logits.shape:', logits.shape)

    #     time = (np.arange(0, np.prod(x.shape) // K).reshape(-1, int(interval * 60 * fs)) / (fs * 60))[seqs]  # (M, T)
    #     if interval != 5:  # files are saved in 5 min sequences
    #     print(seqs)
    x = (x.transpose(1, 0, 2).reshape(K, -1).reshape(K, -1, int(interval * 60 * fs)).transpose(1, 0, 2))[
        seqs
    ]  # (M, K, T)
    time = (time.reshape(-1).reshape(-1, int(interval * 60 * fs)) / (fs * 60))[seqs]  # (M, T)
    #     x = x[seqs]  # (M, K, T)
    #     time = (np.arange(0, np.prod(x.shape) // 5).reshape(-1, 5 * 60 * fs) / (fs * 60))
    #     print('hej')
    M, _, T = x.shape
    print(f"M: {M}, T: {T}")
    #     print(f'x[0, 0, 0:10]: {x[0, 0, 0:10]}')

    #     print(time.shape)
    #     print('x.shape', x.shape)
    seq_idx = int(np.argwhere(seqs == seq_nr).flatten())
    seq_idx_vec = slice(seq_idx - 1, seq_idx + 2)
    print(seq_idx_vec)
    t = time[seq_idx_vec].flatten()
    x = x.transpose(0, 2, 1).reshape(-1, K)  # (M, T, K)  # (MxT, K)
    x = ((x / x.max(axis=0)).reshape(-1, int(interval * 60 * fs), K))[seq_idx_vec]  # (L, K)
    x = x.transpose(2, 0, 1).reshape(5, -1).T
    print("x.shape:", x.shape)
    #     print('t.shape:', t.shape)
    #     print('x.shape:', x.shape)
    #     print('target.shape:', target.shape)
    #     print('pred.shape:', pred.shape)
    #     print('logits.shape:', logits.shape)
    target = (target.transpose(1, 0).reshape(K, -1, int(2 * interval)).transpose(1, 2, 0))[seq_idx_vec]
    pred = (pred.transpose(1, 0).reshape(K, -1, int(2 * interval)).transpose(1, 2, 0))[seq_idx_vec]
    logits = (logits.transpose(1, 0).reshape(K, -1, int(interval * 60)).transpose(1, 2, 0))[seq_idx_vec]  # (M, T, K)
    logits = logits.transpose()
    epoch_nrs = epoch_nrs.reshape(-1, int(interval * 2))[seq_idx_vec]
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
    #     print('n_epochs:', n_epochs)

    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

    # Setup colors
    cmap = np.array(
        [
            [0.4353, 0.8157, 0.9353],  # W
            [0.9490, 0.9333, 0.7725],  # N1
            [0.9490, 0.6078, 0.5118],  # N2
            [0.6863, 0.2078, 0.2784],  # N3
            [0.0000, 0.4549, 0.7373],
        ],  # R
    )
    displacement = np.vstack(
        [
            0 * np.ones(x.shape[0]),
            1 * np.ones(x.shape[0]),
            2 * np.ones(x.shape[0]),
            3 * np.ones(x.shape[0]),
            4 * np.ones(x.shape[0]),
        ]
    ).T

    #     window_dur = 3
    #     delta_f = 1
    #     mts_params = dict(
    #         frequency_range=[0, 20],
    #         time_bandwidth=window_dur * delta_f / 2,
    #         window_params=[window_dur, 1],
    #         min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs))))
    #     )
    #     Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 1], fs, **mts_params, plot_on=True)

    # Setup figure
    fig, axes = plt.subplots(
        figsize=(10, 6),
        nrows=4,
        ncols=1,
        squeeze=True,
        dpi=150,
        gridspec_kw={
            "height_ratios": [3, 3, 1, 1],
            #             'width_ratios': [15, 1],
            #             'wspace': 0.05
        },
    )
    #     if title is not None:
    #         title += f' | Seq. nr. {seq_nr}'

    #     else:
    if title is None:
        title = f"{record_id} | Seq. nr. {seq_nr}"
    fig.suptitle(title)

    # Plot signal data
    current_ax = axes[1]
    current_ax.plot(t, x / x.max() - displacement, "k", linewidth=0.25)
    #     axes[0, 0].plot(t, x - displacement, "k", linewidth=0.15)
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    current_ax.set_xlim(t[0], t[-1])
    current_ax.get_xaxis().set_visible(False)

    # Plot power spectral data
    current_ax = axes[0]
    #     nperseg = x.shape[0] // 8
    #     seg_dur = 3 # seconds
    #     stft_params = dict(
    #         fs=fs,
    #         nperseg=seg_dur * fs,
    #         noverlap=int((seg_dur - 0.1) * fs),
    #         nfft=int(2 ** np.ceil(np.log2(np.abs(seg_dur * fs)))),
    #         detrend=False,
    #         axis=-1,
    #     )
    #     stft_f, stft_t, Zxx = signal.stft(x[:, 0], **stft_params)
    #     current_ax.pcolormesh(stft_t, stft_f[stft_f < 20], 10 * np.log10(np.abs(Zxx[stft_f < 20])), cmap='jet')

    if spectrum_type == "multitaper":
        window_dur = 6
        window_step = 0.1
        delta_f = 1
        mts_params = dict(
            frequency_range=[0, 20],
            time_bandwidth=window_dur * delta_f / 2,
            window_params=[window_dur, window_step],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs)))),
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(x[:, 1], fs, **mts_params, plot_on=False, verbose=verbose)
        print("spec_f.shape:", np.array(spec_f).flatten().shape)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(x[:, 0], signal.morlet2, width, w=w).T

    current_ax.pcolormesh(np.array(spec_t).flatten(), np.array(spec_f).flatten(), nanpow2db(Zxx).T, cmap="jet")

    #     print('nperseg: ', nperseg)
    #     print('nfft: ', nfft)
    #     f, Pxx = signal.welch(x[:, 0], fs=128, nperseg=nperseg, nfft=nfft, average='median')
    #     f, Pxx = signal.csd(x[:, 0], x[:, 1], fs=128, nperseg=nperseg, nfft=nfft, average='mean')
    #     Pxx_norm = Pxx / Pxx.max(axis=0)
    #     Pxx_norm = np.abs(Pxx) / np.abs(Pxx).max()
    #     Pxx_norm = np.abs(Pxx)
    #     Pxx_norm = Pxx
    #     f_displacement = np.vstack(
    #         [0 * np.ones(Pxx.shape[0]), 1 * np.ones(Pxx.shape[0]), 2 * np.ones(Pxx.shape[0]), 3 * np.ones(Pxx.shape[0]), 4 * np.ones(Pxx.shape[0])]
    #     ).T
    #     print(f)
    #     print('f.shape: ', f.shape)
    #     print('Pxx.shape: ', Pxx.shape)
    #     print('Pxx.max(): ', Pxx.max())

    #     gs = axes[0, 1].get_gridspec()
    #     for ax in axes[:, 1]:
    #         ax.remove()
    #     bigax = fig.add_subplot(gs[:, 1])
    #     bigax.yaxis.tick_right()
    #     bigax.yaxis.set_label_position("right")
    #     bigax.semilogx(Pxx_norm[f < 20], f[f < 20], 'k', linewidth=0.2)
    # #     bigax.set_xlim([0.001, 260])
    #     bigax.set_xlim([1e-8, 1e-2])
    # #     bigax.invert_yaxis()
    #     bigax.invert_xaxis()
    #     bigax.set_ylabel('Frequency [Hz]')
    #     bigax.set_xlabel('CSD [uV^2/Hz]')

    #     axes[0, 1].semilogx(f, Pxx_norm - f_displacement, 'k', linewidth=0.2)
    #     axes[0, 1].get_xaxis().set_visible(False)
    #     bigax.set_xlabel('Frequency (Hz)')
    #     bigax.set_xticks([0, 10, 20, 30])
    #     bigax.set_xlim(0, 35)
    #     bigax.set_ylim(0, 0.5)
    #     bigax.get_yaxis().set_visible(False)

    # Add vertical divider lines
    #     current_ax = axes[0, 0]
    current_ax = axes[1]
    vline_coords = np.arange(0, len(hypnogram)) / 2 + t[0]
    for xc, h in zip(vline_coords, hypnogram):
        current_ax.axvline(xc + 0.5, linewidth=0.5, color="grey")

    # Plot 1 s hypnodensity
    #     current_ax = axes[1, 0]
    current_ax = axes[2]
    h = logits.T
    print("h.shape:", h.shape)
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    current_ax.get_xaxis().set_visible(False)
    current_ax.set_xlim([0, len(hypnodensity.T) - 1])
    current_ax.set_ylim([0.0, 1.0])
    current_ax.set_ylabel("1 s")
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.tick_params(axis="both", which="both", length=0)

    # Add vertical divider lines
    vline_coords = np.arange(0, hypnodensity.shape[1] // 30)
    for xc, h in zip(vline_coords, hypnogram):
        current_ax.axvline(xc * 30, linewidth=0.5, color="grey")

    # Plot predicted hypnodensity at 30 s
    #     current_ax = axes[2, 0]
    current_ax = axes[3]
    h = pred.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            current_ax.fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    #     axes[2, 0].get_xaxis().set_visible(False)
    plt.setp(current_ax.get_yticklabels(), visible=False)
    current_ax.set_xlabel("Time (min)")
    current_ax.set_ylabel("30 s")
    current_ax.set_xlim(0, hypnodensity.shape[1] - 1)
    #     print('hypnodensity.shape', hypnodensity.shape)
    current_ax.set_ylim(0.0, 1.0)
    current_ax.tick_params(axis="y", which="both", length=0)
    current_ax.set_xticks(np.arange(0, hypnodensity.shape[1], 4))
    current_ax.set_xticklabels(np.arange(0, hypnodensity.shape[1], 4) // 2)

    # Add vertical divider lines, manual and predicted 30 s hypnograms
    print(epoch_nrs)
    for xc, h_pred, h_true, epch_nr in zip(vline_coords, hypnodensity.argmax(0)[:-1], hypnogram, epoch_nrs):
        current_ax.axvline(xc + 1, linewidth=0.5, color="grey")  # Divider line
        txt = current_ax.text(
            xc + 0.5, 7 - 1.25, hypnogram_dict[h_true], horizontalalignment="center", color=cmap[h_true]
        )  # manual hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="grey")])
        txt = current_ax.text(
            xc + 0.5, 1.075, hypnogram_dict[h_pred], horizontalalignment="center", color=cmap[h_pred]
        )  # automatic hypnogram
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="gray")])
        txt = current_ax.text(xc + 0.5, 2.4, epch_nr, horizontalalignment="center")

    #     # Add manual and predicted 30 s hypnograms
    #     for xc, h in zip(vline_coords, hypnodensity.argmax(0)[:-1]):

    # Add predicted 30 s hypnogram at the bottom
    #     vline_coords = np.arange(0, hypnodensity.shape[1] - 1)
    #     print(vline_coords)
    #     for xc, h in zip(vline_coords, hypnodensity.argmax(0)[:-1]):

    # Add text objects
    current_ax.text(-0.2, 7 - 1.25, "Manual", ha="right", color="grey")
    current_ax.text(-0.2, 1.075, "Automatic", ha="right", color="grey")

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
        fig.savefig(f"results/{save_path}", dpi=300, bbox_inches="tight", pad_inches=0)


#     plt.show()
    plt.close()


def plot_hypnodensity(logits, preds, trues, title=None, save_path=None):

    # Setup title
    f, ax = plt.subplots(nrows=4, figsize=(20, 5), dpi=400)
    f.suptitle(title)

    # Setup colors
    cmap = np.array(
        [
            [0.4353, 0.8157, 0.9353],  # W
            [0.9490, 0.9333, 0.7725],  # N1
            [0.9490, 0.6078, 0.5118],  # N2
            [0.6863, 0.2078, 0.2784],  # N3
            [0.0000, 0.4549, 0.7373],
        ],  # R
    )

    # Plot the hypnodensity
    h = logits.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            ax[0].fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    ax[0].get_xaxis().set_visible(False)
    ax[0].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[0].set_ylim(0.0, 1.0)
    ax[0].set_ylabel("1 s")
    plt.setp(ax[0].get_yticklabels(), visible=False)
    ax[0].tick_params(axis="both", which="both", length=0)

    # Create legend
    legend_elements = [
        mpl.patches.Patch(facecolor=cm, edgecolor=cm, label=lbl)
        for cm, lbl in zip(cmap, ["W", "N1", "N2", "N3", "REM"])
    ]
    ax[0].legend(handles=legend_elements, loc="lower center", bbox_to_anchor=[0.5, 1.0], ncol=5)
    #     sns.despine(top=True, bottom=True, left=True, right=True)
    #     plt.tight_layout()

    # Plot predicted hypnodensity at 30 s
    h = preds.T
    hypnodensity = np.concatenate([h, h[:, -1, np.newaxis]], axis=-1)
    y_ = np.zeros((hypnodensity.shape[0] + 1, hypnodensity.shape[1]))
    y_[1:, :] = np.cumsum(hypnodensity, axis=0)
    l = []
    for n in range(hypnodensity.shape[0]):
        l.append(
            ax[1].fill_between(
                np.arange(hypnodensity.shape[1]),
                y_[n, :],
                y_[n + 1, :],
                edgecolor="face",
                facecolor=cmap[n, :],
                linewidth=0.0,
                step="post",
            )
        )
    ax[1].get_xaxis().set_visible(False)
    ax[1].set_xlim(0, hypnodensity.shape[1] - 1)
    ax[1].set_ylim(0.0, 1.0)
    ax[1].set_ylabel("30 s")
    plt.setp(ax[1].get_yticklabels(), visible=False)
    ax[1].tick_params(axis="both", which="both", length=0)

    # Plot predicted hyponogram
    ax[2].plot(preds.argmax(axis=-1))
    ax[2].set_xlim(0, trues.shape[0] - 1)
    ax[2].set_ylim(-0.5, 4.5)
    ax[2].get_xaxis().set_visible(False)
    ax[2].set_yticks([0, 1, 2, 3, 4])
    ax[2].set_yticklabels(["W", "N1", "N2", "N3", "R"])
    ax[2].set_ylabel("Automatic")

    # Plot true hyponogram
    ax[3].plot(trues.argmax(axis=-1))
    ax[3].set_xlim(0, trues.shape[0] - 1)
    ax[3].set_ylim(-0.5, 4.5)
    ax[3].set_yticks([0, 1, 2, 3, 4])
    ax[3].set_yticklabels(["W", "N1", "N2", "N3", "R"])
    ax[3].set_xticks(np.arange(0, trues.shape[0] - 1, 20))
    ax[3].set_xticklabels(np.arange(0, trues.shape[0] - 1, 20) * 30 // 60)
    ax[3].set_xlabel("Time (min)")
    ax[3].set_ylabel("Manual")

    # Save figure
    if save_path is not None:
        f.savefig(f"results/{save_path}", dpi=300, bbox_inches="tight", pad_inches=0)
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


def plot_data(
    X,
    y,
    current_file=None,
    target_resolution=1,
    output_resolution=30,
    seq_nr=None,
    fs=128,
    spectrum_type="multitaper",
    title=None,
    save_path=None,
    verbose=False,
):
    """
    Args:
        X (ndarray): shape (N * 30 * fs, C)
        y (ndarray): shape (N, K) onehot encoded
    Note:
        N: number of epochs (30 s)
        fs: sampling frequency
        C: number of channels
        K: number of classes
    """

    print(f"Current file: {current_file}")
    print(f"Selected sequence: {seq_nr}")

    N, K = X.shape
    t = np.arange(0, N) / fs

    if y.shape[-1] != K:
        y = y.transpose()
    hypnogram = y.argmax(axis=-1)[:: output_resolution // target_resolution]
    n_epochs = len(hypnogram)
    hypnogram_dict = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
    displacement = np.vstack([0 * np.ones(N), 1 * np.ones(N), 2 * np.ones(N), 3 * np.ones(N), 4 * np.ones(N)]).T

    # Setup figure
    fig, axes = plt.subplots(figsize=(15, 5), nrows=3, ncols=1, squeeze=True, dpi=150,)
    # if title is None:
    #     title = f"{record_id} | Seq. nr. {seq_nr}"
    # fig.suptitle(title)

    # Plot power spectral data
    current_ax = axes[0]
    if spectrum_type == "multitaper":
        window_dur = 3
        window_step = 0.1
        delta_f = 1.5
        mts_params = dict(
            frequency_range=[0, 20],
            time_bandwidth=window_dur * delta_f / 2,
            window_params=[window_dur, window_step],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs)))),
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(X[:, 0], fs, **mts_params, plot_on=False, verbose=verbose)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(X[:, 1], signal.morlet2, width, w=w).T
    current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto"
    )
    current_ax.get_xaxis().set_visible(False)
    current_ax = axes[1]
    if spectrum_type == "multitaper":
        window_dur = 3
        window_step = 0.1
        delta_f = 1.5
        mts_params = dict(
            frequency_range=[0, 20],
            time_bandwidth=window_dur * delta_f / 2,
            window_params=[window_dur, window_step],
            min_nfft=int(2 ** np.ceil(np.log2(np.abs(window_dur * fs)))),
        )
        Zxx, spec_t, spec_f = multitaper_spectrogram(X[:, 1], fs, **mts_params, plot_on=False, verbose=verbose)
    elif spectrum_type == "cwt":
        spec_t = np.arange(0, 3 * 30 * fs) / fs
        w = 20.0
        spec_f = np.linspace(0, 20, 100)
        width = w * fs / (2 * spec_f * np.pi)
        Zxx = signal.cwt(X[:, 1], signal.morlet2, width, w=w).T
    current_ax.pcolormesh(
        np.array(spec_t).flatten(), np.array(spec_f).flatten(), nanpow2db(np.abs(Zxx)).T, cmap="jet", shading="auto"
    )
    current_ax.get_xaxis().set_visible(False)

    # Plot signal data
    current_ax = axes[2]
    current_ax.plot(
        t,
        # (X - X.mean(axis=0, keepdims=True)) / X.std(axis=0, keepdims=True) - displacement,
        X / X.max(axis=0) - displacement,
        "k",
        linewidth=0.25,
    )
    current_ax.set_xlim(t[0], t[-1])
    current_ax.set_xticks(np.arange(0, N // fs, 15))
    current_ax.set_xticklabels(np.arange(0, N // fs, 15))
    current_ax.set_yticks([0, -1, -2, -3, -4])
    current_ax.set_yticklabels(["EEG C", "EEG O", "EOG L", "EOG R", "EMG"])
    current_ax.set_ylim([-5, 1])
    # current_ax.get_xaxis().set_visible(False)

    # Add vertical divider lines
    current_ax = axes[2]
    vline_coords = np.arange(0, len(hypnogram)) * 30 + t[0]
    for xc, h in zip(vline_coords, hypnogram):
        current_ax.axvline(xc, linewidth=0.5, color="grey")

    # # Add vertical divider lines, manual and predicted 30 s hypnograms
    # manual_str_placement = 8.8
    # for xc, h_pred, h_true, epch_nr in zip(vline_coords, hypnodensity.argmax(0)[:-1], hypnogram, epoch_nrs):
    #     current_ax.axvline(xc + 1, linewidth=0.5, color="grey")  # Divider line
    #     txt = current_ax.text(
    #         xc + 0.5, manual_str_placement, hypnogram_dict[h_true], horizontalalignment="center", color=cmap[h_true]
    #     )  # manual hypnogram
    #     txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="grey")])
    #     txt = current_ax.text(
    #         xc + 0.5, auto_str_placement, hypnogram_dict[h_pred], horizontalalignment="center", color=cmap[h_pred]
    #     )  # automatic hypnogram
    #     txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground="gray")])
    # #         txt = current_ax.text(xc + 0.5, 2.4, epch_nr, horizontalalignment="center")

    # # Add text objects
    # current_ax.text(0.0, manual_str_placement, "Manual", ha="left", color="grey")
    # current_ax.text(0.0, auto_str_placement, "Automatic", ha="left", color="grey")
    # #     current_ax.text(-0.025, 1.075, 'Automatic', ha='right', color='grey')

    # # Save figure
    if save_path is not None:
        fig.savefig(f"results/{save_path}", dpi=300, bbox_inches="tight", pad_inches=0)
    else:
        fig.savefig("results/test.png", dpi=300, bbox_inches="tight", pad_inches=0)
