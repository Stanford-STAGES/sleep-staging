import os
import pickle
import sys

import numpy as np
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

import datasets
import utils

torch.backends.cudnn.benchmark = True


def run_predict():

    args = utils.get_args("predict")

    # If you wish to view applied settings, uncomment these two lines.
    # pprint.pprint(vars(args))

    # Remember to seed!
    if args.model_type == "stages":
        seed = int(args.model_name.split("_")[-1])
    else:
        seed = 1337
    seed_everything(seed)

    # Setup model
    model = utils.get_model(args)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, logger=False, deterministic=True)

    # ------------------------------------------------------------------------------- #
    # TEST ON NEW DATA
    # ------------------------------------------------------------------------------- #
    results_dir = os.path.join(os.path.dirname(args.resume_from_checkpoint), "predictions")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ds_args = model.hparams.copy()
    if args.predict_on:
        test_dm = []
        test_args = ds_args.copy()
        test_args.update({k[2:]: vars(args)[k[2:]] for k in sys.argv[1::2]})
        test_args["balanced_sampling"] = False
        test_args["batch_size"] = 1
        test_args["sequence_length"] = "full"
        test_args["adjustment"] = 0
        for cohort, data_path in args.predict_on.items():
            test_dm.append((cohort, datasets.BaseDataModule(data_dir={"train": None, "test": data_path}, **test_args)),)
        for dm in test_dm:
            dm[1].setup("test")
    else:
        ds_args.update({k[2:]: vars(args)[k[2:]] for k in sys.argv[1::2]})
        ds_args["balanced_sampling"] = False  # This should not be set on eval or test data
        ds_args["batch_size"] = 1
        ds_args["sequence_length"] = "full"
        test_dm = []
        test_dm.append(("test", datasets.SscWscDataModule(**ds_args)),)
        test_dm[-1][1].setup("test")

        test_args = ds_args.copy()
        test_args.pop("data_dir")
        for cohort_name in ["ahc", "dhc", "ihc", "jcts", "khc"]:
            test_dm.append(
                (cohort_name, datasets.BaseDataModule(data_dir={"train": None, "test": f"data/{cohort_name}/raw"}, **test_args)),
            )
            test_dm[-1][1].setup("test")

    for name, tdm in test_dm:

        if tdm.has_setup_fit:
            predictions = trainer.test(model, test_dataloaders=tdm.val_dataloader(), verbose=False)
        elif tdm.has_setup_test:
            predictions = trainer.test(model, test_dataloaders=tdm.test_dataloader(), verbose=False)
        else:
            raise AttributeError

        if not isinstance(predictions, dict):
            predictions = {}

        if not model.use_ddp or (model.use_ddp and torch.distributed.get_rank() == 0):

            for filepath in sorted(os.listdir(os.path.join(results_dir, name))):
                with open(os.path.join(results_dir, name, filepath), "rb") as pkl:
                    prediction = pickle.load(pkl)
                predictions.update(
                    {
                        prediction["record"]: {
                            "predicted": prediction["predicted"],
                            "true": prediction["true"],
                            "stable_sleep": prediction["stable_sleep"],
                        }
                    }
                )

            df, cm_sub, cm_tot = utils.evaluate_performance(predictions, evaluation_windows=[1], cases=["all",],)
            with np.printoptions(precision=3, suppress=True):
                s = ""
                for eval_window in cm_tot.keys():
                    s += "\n"
                    s += f"{name.upper()}\n"
                    s += "---------------------------------\n"
                    for case in cm_tot[eval_window].keys():
                        df_ = df.query(f'Window == "{eval_window} s" and Case == "{case}"')
                        s += f"Case: {case}\n"
                        s += f"{cm_tot[eval_window][case]}\n"
                        NP = cm_tot[eval_window][case].sum(axis=1)
                        PP = cm_tot[eval_window][case].sum(axis=0)
                        N = cm_tot[eval_window][case].sum()
                        precision = np.diag(cm_tot[eval_window][case]) / (PP + 1e-10)
                        recall = np.diag(cm_tot[eval_window][case]) / (NP + 1e-10)
                        f1 = 2 * precision * recall / (precision + recall + 1e-10)
                        acc = np.diag(cm_tot[eval_window][case]).sum() / N

                        pe = N ** (-2) * (NP @ PP)
                        kappa = 1 - (1 - acc) / (1 - pe)

                        c = np.diag(cm_tot[eval_window][case]).sum()
                        mcc = (c * N - NP @ PP) / (np.sqrt(N ** 2 - (PP @ PP)) * np.sqrt(N ** 2 - (NP @ NP)))

                        s += "\n"
                        s += f'Precision:\t{df_["Precision"].mean():.3f} +/- {df_["Precision"].std():.3f} \t|\t{precision}\n'
                        s += f'Recall:\t\t{df_["Recall"].mean():.3f} +/- {df_["Recall"].std():.3f} \t|\t{recall}\n'
                        s += f'F1: \t\t{df_["F1"].mean():.3f} +/- {df_["F1"].std():.3f} \t|\t{f1}\n'
                        s += f'Accuracy:\t{df_["Accuracy"].mean():.3f} +/- {df_["Accuracy"].std():.3f} \t|\t{acc:.3f}\n'
                        s += f'Kappa:\t\t{df_["Kappa"].mean():.3f} +/- {df_["Kappa"].std():.3f} \t|\t{kappa:.3f}\n'
                        s += f'MCC:\t\t{df_["MCC"].mean():.3f} +/- {df_["MCC"].std():.3f} \t|\t{mcc:.3f}\n'
                        s += "\n"
            print(s)
            with open(os.path.join(results_dir, f"{name}_results.txt"), "w") as txt_file:
                print(s, file=txt_file)
            df.to_csv(os.path.join(results_dir, f"{name}_results.csv"))
            with open(os.path.join(results_dir, f"{name}_confusionmatrix.pkl"), "wb") as pkl:
                pickle.dump(
                    {"confusionmatrix_subject": cm_sub, "confusionmatrix_total": cm_tot,}, pkl,
                )

    return 0


if __name__ == "__main__":

    run_predict()
