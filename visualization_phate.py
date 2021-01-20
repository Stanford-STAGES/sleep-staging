import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scprep
import seaborn as sns
from tqdm import tqdm


os.chdir("/home/users/alexno/sleep-staging")
sns.set(context="paper", rc={"figure.figsize": (14, 10)})
HYP_DICT = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}
logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.DEBUG)


def main():

    logging.info("Loading master data file.")
    df = pd.read_csv("overview_file_cohortsEM-ling1.csv")

    selected_experiment = "experiments/massc/avg_kw21/20201126_043826/SSC-WSC_test_predictions.pkl"
    logging.info(f'Loading predictions file: "{selected_experiment}"')
    with open(selected_experiment, "rb") as f:
        predictions = pickle.load(f)

    logging.info("Loading associated prediction dataframe")
    df_preds = pd.read_csv(os.path.join(os.path.dirname(selected_experiment), "SSC-WSC_test_results.csv"), index_col=0)
    df_preds = pd.merge(
        left=df_preds, right=df[["ID", "Diagnosis", "Label"]], how="left", right_on="ID", left_on="SubjectID"
    )

    logging.info(f"Unique classes: {df_preds['Diagnosis'].unique()}")
    logging.info(f"Window unique classes: {df_preds['Window'].unique()}")
    logging.info(f"Case unique classes: {df_preds['Case'].unique()}")

    # get dataframe sorted by accuracy in each diagnosis group
    g = (
        df_preds.groupby(["Diagnosis"])
        .apply(lambda x: x.sort_values(["Accuracy"], ascending=False))
        .reset_index(drop=True)
    )
    # select top 7 rows within each continent
    topK = 7
    logging.info(f"Get topK={topK} subjects from each group.")
    selected_top10 = g.groupby("Diagnosis").head(topK).reset_index(drop=True)
    selected_top10["Diagnosis"] = selected_top10["Diagnosis"].str.replace("'", "")

    # Select only controls and NT1
    selected_top10 = selected_top10.loc[selected_top10["Diagnosis"].str.contains("NARC")]
    print("")
    X = []
    y = []
    l = []
    d = []
    for idx, selected_record in tqdm(selected_top10.iterrows(), desc="Processing records..."):
        fileid = selected_record["FileID"]
        diagnosis = selected_record["Diagnosis"].replace("'", "")
        record = predictions[fileid]
        logits = record["logits"]
        target = record["true"].argmax(axis=1).repeat(30)
        labels = np.array([HYP_DICT[t] for t in target])
        labels_disease = [diagnosis for _ in range(len(target))]

        X.append(logits)
        y.append(target)
        l.append(labels)
        d.extend(labels_disease)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    l = np.concatenate(l, axis=0)
    # X = np.asarray(X)
    # y = np.array(y)
    # l = np.array(l)

    print("")
    for n_knn in tqdm([525, 50, 100, 150, 200, 250], desc="Plotting PHATE..."):
        phate_op = phate.PHATE(n_jobs=-1, random_state=1337, knn=n_knn)
        y_phate = phate_op.fit_transform(X)
        title = f"KNN: {n_knn}"
        scprep.plot.scatter2d(
            y_phate,
            c=d,
            cmap={"NON-NARCOLEPSY CONTROL": "blue", "T1 NARCOLEPSY": "red"},
            ticks=False,
            label_prefix="PHATE",
        )
        plt.title(title)
        plt.savefig(
            f"results/phate/phate_subjects{topK:03}_knn{n_knn:03}_disorder.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )
        scprep.plot.scatter2d(
            y_phate, c=l, ticks=False, label_prefix="PHATE", cmap="Spectral",
        )
        plt.title(title)
        plt.savefig(
            f"results/phate/phate_subjects{topK:03}_knn{n_knn:03}_stages.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
        )


if __name__ == "__main__":
    main()
