import os
import pprint
import pickle
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer

# from pytorch_lightning.callbacks import LearningRateLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.profiler import AdvancedProfiler
# from torch.utils.data import DataLoader

import datasets
import models
import utils

# from datasets import ISRCDataset, KoreanDataset, SscWscPsgDataset
# from models import *
# from utils import evaluate_performance

torch.backends.cudnn.benchmark = True


# def main(args):


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

    ds_args = model.hparams
    if args.predict_on:
        test_dm = []
        test_args = dict(
            batch_size=1,
            n_workers=args.n_workers,
            n_records=10,
            scaling="robust",
            adjustment=0,
            n_jobs=ds_args.n_jobs,
            sequence_length="full",
            n_channels=ds_args.n_channels,
            balanced_sampling=False,
        )
        # test_dm.append(("DHC", datasets.BaseDataModule(data_dir={"train": None, "test": "data/dhc/raw"}, **test_args)),)
        # test_dm.append(("IHC", datasets.BaseDataModule(data_dir={"train": None, "test": "data/ihc/raw"}, **test_args)),)
        for cohort, data_path in args.predict_on.items():
            test_dm.append((cohort, datasets.BaseDataModule(data_dir={"train": None, "test": data_path}, **test_args)),)
        # test_dm.append(("JCTS", datasets.BaseDataModule(data_dir={"train": None, "test": "data/jcts/raw"}, **test_args)),)
        # test_dm.append(("AHC", datasets.BaseDataModule(data_dir={"train": None, "test": "data/ahc/raw"}, **test_args)),)
        for dm in test_dm:
            dm[1].setup("test")
    else:
        ds_args["balanced_sampling"] = False  # This should not be set on eval data
        ds_args["batch_size"] = 1
        ds_args["sequence_length"] = "full"
        ds_args["n_workers"] = args.n_workers
        # ds_args["data_dir"] = "data/train"
        # ds_args["n_jobs"] = args.n_jobs
        # ds_args["n_records"] = None
        # ds_args["batch_size"] = args.batch_size
        # ds_args["limit_test_batches"] = args.limit_test_batches
        # ds_args["adjustment"] = args.adjustment
        # ds_args["sequence_length"] = args.sequence_length

        test_dm = ("eval", datasets.SscWscDataModule(**ds_args))
        test_dm[1].setup("fit")

        # test_dm.append(("SSC-WSC_test", datasets.SscWscDataModule(**ds_args)),)
        # test_dm[-1][1].setup("test")

        # test_dm.append(("SSC-WSC_more-spindles", datasets.SscWscDataModule(**ds_args)),)
        # test_dm[-1][1].setup("test")
        # khc_args = datasets.KHCDataModule.add_dataset_specific_args(ArgumentParser()).parse_known_args()[0]
        # test_dm.append(("KHC", datasets.KHCDataModule(**vars(khc_args))),)
        # test_dm[-1][1].setup("test")

    for name, tdm in test_dm:
        # predictions = trainer.test(model, test_dataloaders=tdl, verbose=False)[0]
        # predictions = trainer.test(model, datamodule=tdm, verbose=False)

        if tdm.has_setup_fit:
            predictions = trainer.test(model, test_dataloaders=tdm.val_dataloader(), verbose=False)
        elif tdm.has_setup_test:
            predictions = trainer.test(model, test_dataloaders=tdm.test_dataloader(), verbose=False)
        else:
            raise AttributeError

        # trainer.test(model, datamodule=tdm, verbose=False)
        if not model.use_ddp or (model.use_ddp and torch.distributed.get_rank() == 0):
            # with open(os.path.join(results_dir, f"{name}_predictions.pkl"), "rb") as pkl:
            #     predictions = pickle.load(pkl)
            predictions = predictions[0]

            with open(os.path.join(results_dir, f"{name.lower()}_predictions.pkl"), "wb") as pkl:
                pickle.dump(predictions, pkl)

            os.makedirs(os.path.join(results_dir, "predictions", f"{name.lower()}"), exist_ok=True)
            for record, record_predictions in tqdm(predictions.items()):
                with open(
                    os.path.join(results_dir, "predictions", f"{name.lower()}", f"preds_{record.split('.')[0]}.pkl"), "wb",
                ) as pkl:
                    pickle.dump(record_predictions, pkl)

            df, cm_sub, cm_tot = utils.evaluate_performance(
                predictions,
                evaluation_windows=[1],
                cases=[
                    "all",
                    # "stable",
                ],
            )
            with np.printoptions(precision=3, suppress=True):
                s = ""
                for eval_window in cm_tot.keys():
                    # print()
                    s += "\n"
                    s += f"Evaluation window - {eval_window} s\n"
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
                        # print(f"F1: {f1}")
                        # print()
            print(s)
            with open(os.path.join(results_dir, f"{name}_results.txt"), "w") as txt_file:
                print(s, file=txt_file)
            # pprint.pprint(cm_tot[1])
            # pprint.pprint(cm_tot[30])
            df.to_csv(os.path.join(results_dir, f"{name}_results.csv"))
            with open(os.path.join(results_dir, f"{name}_confusionmatrix.pkl"), "wb") as pkl:
                pickle.dump(
                    {"confusionmatrix_subject": cm_sub, "confusionmatrix_total": cm_tot,}, pkl,
                )

    # def run_testing(_data, _name):
    #     predictions = trainer.test(model, test_dataloaders=_data, verbose=False)
    #     with open(os.path.join(results_dir, f"{_name}_predictions.pkl"), "wb") as pkl:
    #         pickle.dump(predictions, pkl)

    #     df_results = evaluate_performance(predictions)
    #     df_results.to_csv(os.path.join(results_dir, f"{_name}_results.csv"))

    # Run predictions on SSC-WSC test data
    # model.dataset_params["n_records"] = 10
    # test_data = DataLoader(SscWscPsgDataset(**model.dataset_params), **test_params)
    # predictions = trainer.test(model, test_dataloaders=test_data, verbose=False)
    # # test_data = DataLoader(SscWscPsgDataset("./data/test/raw/ssc_wsc", overlap=False, n_records=10), **test_params)
    # run_testing(test_data, "SSC-WSC")

    # # Test on KHC data
    # khc_data = DataLoader(KoreanDataset(overlap=False, n_records=10), **test_params)
    # run_testing(khc_data, "KHC")

    # predictions = trainer.test(model, test_dataloaders=test_data, verbose=False)[0]
    # results = evaluate_performance(predictions)
    # with open(os.path.join(results_dir, "SSC-WSC_predictions.pkl"), "wb") as pkl:
    #     pickle.dump(predictions, pkl)
    # results.to_csv(os.path.join(results_dir, "SSC-WSC_results.csv"))

    # # KHC data
    # khc_data = DataLoader(KoreanDataset(), num_workers=args.n_workers, pin_memory=True)
    # results = trainer.test(test_dataloaders=khc_data, verbose=False)
    # df = evaluate_performance(results)
    # print(len(results.keys()))
    # with open(os.path.join(results_dir, "KHC.pkl"), "wb") as pkl:
    #     pickle.dump(results, pkl)

    # df.to_csv(os.path.join(results_dir, 'KHC.csv'))

    return 0


if __name__ == "__main__":

    run_predict()

    # args = utils.get_args()

    # if args.model_type in ["massc", "massc_v2"]:
    # args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
    # args.save_dir.mkdir(parents=True, exist_ok=True)

    # main(args)
