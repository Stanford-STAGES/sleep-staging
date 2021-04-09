import argparse
import logging
import os
import pickle

os.chdir(os.path.expandvars("$HOME/sleep-staging"))

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scprep
import seaborn as sns
from tqdm.notebook import tqdm

sns.set(context="paper", rc={"figure.figsize": (11, 10)})

HYP_DICT = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
logger = logging.getLogger()

df = pd.read_csv("overview_file_cohortsEM-ling1.csv")


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""

    def __init__(self, X_sub, X, labels_sub, labels, name, status, resolution, num_points=5):
        self.xy = X_sub
        self.XY = X
        self.N = self.xy.shape[0]
        self.labels_sub = labels_sub
        self.labels = labels
        self.name = name
        self.resolution = resolution
        self.status = status
        self.num_points = num_points
        # self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(dpi=120)
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(
            self.fig, self.update, frames=self.N, interval=500, init_func=self.setup_plot, blit=True
        )

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        # x, y, s, c = next(self.stream).T
        x, y = self.XY.T
        c = self.labels
        s = 5 * np.ones(len(c))
        self.scat = self.ax.scatter(x, y, c=c, s=s, vmin=0, vmax=4, cmap="Spectral", edgecolor="k")
        self.ax.axis([-0.05, 0.05, -0.05, 0.05])
        handles, labels = self.scat.legend_elements()
        self.legend = self.ax.legend(handles, ["W", "N1", "N2", "N3", "R"])
        self.ax.add_artist(self.legend)
        self.fig.suptitle(f"{self.status}\n{self.resolution} s per dot")
        # self.scat.legend(["W", "N1", "N2", "N3", "R"])
        # self.fig.legend()
        self.ax.set_xlabel("PHATE1")
        self.ax.set_ylabel("PHATE2")
        # self.fig.suptitle("Frame no.: ")
        #         self.scat.title('Frame no.: ')
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (self.scat,)

    def data_stream(self, i):
        """Generate a data stream using num_points points. Data is scaled according to recency."""

        s = 0.5 * np.ones(self.num_points)

        if i < self.num_points:
            extract = slice(0, self.num_points)
            s[i] = 2
        else:
            extract = slice(i - self.num_points + 1, i + 1)
            s[-1] = 2
        xy = self.xy[extract, :]
        c = self.labels_sub[extract]

        return np.c_[xy[:, 0], xy[:, 1], s, c]

    def update(self, i):
        """Update the scatter plot."""
        # data = next(self.stream)
        data = self.data_stream(i)
        # Set x and y data...
        self.scat.set_offsets(data[:, :2])
        # Set sizes...
        self.scat.set_sizes(300 * abs(data[:, 2]) ** 1.5 + 100)
        # Set colors..
        self.scat.set_array(data[:, 3])
        # if i < self.N:
        # Set frame title
        self.fig.suptitle(f"{self.status}\n{self.resolution} s per dot\nFrame no.: {i}")

        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return (self.scat,)

    def save_gif(self, filename):
        filename = f"{os.path.splitext(filename)[0]}_res{self.resolution:02}.gif"
        logger.info(f"\tSaving at: {filename}")
        writergif = animation.PillowWriter(fps=15)
        #         writergif = 'imagemagick'
        self.ani.save(filename, writer=writergif)

    def save_mp4(self, filename):
        filename = f"{os.path.splitext(filename)[0]}_res{self.resolution:02}.mp4"
        logger.info(f"\tSaving at: {filename}")
        writer = animation.FFMpegWriter(fps=24)
        self.ani.save(filename, writer=writer)


def run_cli(args):

    n_knn = args.n_knn
    gamma = args.gamma
    savepath = args.savepath
    resolution = args.resolution
    topK = args.topk
    subject = args.subject
    repeat_factor = 30 // resolution

    logger.info("PHATE input arguments:")
    logger.info(f"\tNumber of neighbors: {n_knn}")
    logger.info(f"\tGamma: {gamma}")
    logger.info(f"\tSleep scoring resolution: {resolution} s")
    logger.info(f"\ttopK: {topK}")
    logger.info(f"\tRepeat factor: {repeat_factor}")
    logger.info(f"\tSaving figures at: {savepath}\n")
    logger.info(f"\tSubject number: {subject}")

    selected_experiment = "experiments/massc/avg_kw21/20201126_043826/SSC-WSC_test_predictions.pkl"
    with open(selected_experiment, "rb") as f:
        predictions = pickle.load(f)
    list_studies = list(predictions.keys())
    df_preds = pd.read_csv(os.path.join(os.path.dirname(selected_experiment), "SSC-WSC_test_results.csv"), index_col=0)
    df_preds = pd.merge(
        left=df_preds, right=df[["ID", "Diagnosis", "Label"]], how="left", right_on="ID", left_on="SubjectID"
    )
    logger.info(f"Unique classes: {df_preds['Diagnosis'].unique()}")
    logger.info(f"Window unique classes: {df_preds['Window'].unique()}")
    logger.info(f"Case unique classes: {df_preds['Case'].unique()}")

    logger.info(f"Selecting {topK} controls...")
    topK_controls = pd.concat(
        [
            df_preds[
                (df_preds["Label"] == 1)
                & (df_preds["Support - W"] > 20)
                & (df_preds["Support - N1"] > 20)
                & (df_preds["Support - N2"] > 20)
                & (df_preds["Support - N3"] > 20)
                & (df_preds["Support - REM"] > 20)
            ]
            .sort_values(["Accuracy"], ascending=False)
            .head(topK // 2)
            .reset_index(drop=True),
            df_preds[
                (df_preds["Label"] == 0)
                & (df_preds["Support - W"] > 20)
                & (df_preds["Support - N1"] > 20)
                & (df_preds["Support - N2"] > 20)
                & (df_preds["Support - N3"] > 20)
                & (df_preds["Support - REM"] > 20)
            ]
            .sort_values(["Accuracy"], ascending=False)
            .head(topK // 2)
            .reset_index(drop=True),
        ]
    )

    X = []
    t = []
    p = []
    l = []
    d = []
    fid = []
    idx = []
    idx_5min = []
    fid_vec = []
    fid_counter = 0

    for idx, selected_record in topK_controls.iterrows():

        fileid = selected_record["FileID"]
        diagnosis = selected_record["Diagnosis"].replace("'", "")
        #         record = predictions[fileid]
        with open(
            os.path.join(
                os.path.dirname(selected_experiment),
                "predictions",
                "ssc-wsc_test",
                f'preds_{fileid.split(".")[0]}.pkl',
            ),
            "rb",
        ) as pkl:
            record = pickle.load(pkl)
        mask = record["true"].sum(axis=1) == 1
        target = (record["true"][mask].argmax(axis=1)).repeat(repeat_factor)
        labels = np.array([HYP_DICT[t] for t in target])
        labels_disease = [diagnosis for _ in range(len(target))]
        seq_nr = record["seq_nr"].repeat(10)[mask].repeat(repeat_factor)  # each 30 s is part of a 5 min sequence
        if resolution == 1:
            logits = record["logits"][mask.repeat(30)]
        elif resolution in [3, 5, 10, 15]:
            try:
                logits = record[f"yhat_{resolution}s"][mask.repeat(repeat_factor)]
            except:
                logger.info(f"Available resolutions: {record.keys()}")
                return
        elif resolution == 30:
            logits = record["predicted"][mask]
        predicted_class = np.array([t for t in logits.argmax(axis=1)])

        X.append(logits)
        t.append(target)
        l.append(labels)
        d.extend(labels_disease)
        fid.extend([fileid for _ in range(len(target))])
        fid_vec.extend([fid_counter for _ in range(len(target))])
        p.extend(predicted_class)
        fid_counter += 1

    X = np.concatenate(X, axis=0)
    y = np.concatenate(t, axis=0)
    l = np.concatenate(l, axis=0)
    p = np.array(p)
    data = pd.DataFrame({"Stage": l, "FileID": fid, "Pred_stage": p, "Status": d})

    logging.info("Creating PHATE object")
    phate_op = phate.PHATE(n_components=2, n_jobs=-1, random_state=1337, knn=n_knn, gamma=gamma)
    X_phate = phate_op.fit_transform(X)

    #     logger.info(f'{data["FileID"][0]}')
    #     logger.info(f"X.shape: {X_phate[data['FileID'] == data['FileID'][0]].shape}")
    #     logger.info(f"p.shape: {p.shape}")
    #     logger.info(f"p: {p[data['FileID'] == data['FileID'][0]]}")

    unique_subjects = data["FileID"].unique()
    if subject:
        sub = unique_subjects[subject]
        anim = AnimatedScatter(
            X_phate[data["FileID"] == sub],
            X_phate,
            p[data["FileID"] == sub],
            p,
            sub,
            data.query(f'FileID == "{sub}"')["Status"].unique()[0],
            resolution,
        )
        # anim.save_gif(f"./results/anim_phate/{sub}")
        anim.save_mp4(f"./results/anim_phate/{sub}")
    else:
        for sub in tqdm(unique_subjects):
            anim = AnimatedScatter(
                X_phate[data["FileID"] == sub],
                X_phate,
                p[data["FileID"] == sub],
                p,
                sub,
                data.query(f'FileID == "{sub}"')["Status"].unique()[0],
                resolution,
            )
            # anim.save_gif(f"./results/anim_phate/{sub}")
            anim.save_mp4(f"./results/anim_phate/{sub}")
    return 0
    # print(X_phate[data["FileID"] == data["FileID"][1]])
    # return X_phate, data

    logging.info("Plotting embedded data as animation")
    ax = scprep.plot.scatter2d(
        X_phate, c=l, cmap="Spectral", ticks=False, label_prefix="PHATE", dpi=300, s=5, fontsize=20
    )
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.text(
        0.99,
        0.01,
        f"PHATE: n_knn={n_knn}, gamma={gamma}",
        transform=ax.transAxes,
        horizontalalignment="right",
        color="k",
        fontsize=20,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_knn", type=int)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--savepath", default=None, type=str)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--subject", default=None, type=int)
    args = parser.parse_args()

    run_cli(args)
