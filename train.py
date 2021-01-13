import os
import pickle

# from argparse import ArgumentParser
# from datetime import datetime
# from pathlib import Path

import numpy as np

# import pandas as pd
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

# from pytorch_lightning.callbacks import LearningRateMonitor
# from pytorch_lightning.callbacks import ModelCheckpoint

# from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.profiler import AdvancedProfiler
# from torch.utils.data import DataLoader

import datasets
import utils

# from datasets import *
# from models import *
# from utils import *

torch.backends.cudnn.benchmark = True


# def main(args):
def run_training():

    args = utils.get_args()

    # Remember to seed!
    seed_everything(args.seed)

    # Setup data module for training
    dm, args = utils.get_data(args)
    # if args.model_type == "stages":
    #     dm = datasets.STAGESDataModule(**vars(args))
    #     dm.setup()
    # else:
    #     dm = datasets.SscWscDataModule(**vars(args))
    #     dm.setup()
    #     args.cb_weights = dm.train.dataset.cb_weights
    # return

    # Setup model
    model = utils.get_model(args)

    # Setup callbacks
    loggers, callbacks = utils.get_loggers_callbacks(args, model)
    # # wandb_logger_params = dict(save_dir="experiments", project="sleep-staging", log_model=True)
    # wandb_logger_params = dict(save_dir=args.save_dir, project="sleep-staging", log_model=False)
    # try:
    #     csv_logger_params = dict(save_dir="./experiments", version=args.save_dir.parts[-1])
    # except AttributeError:
    #     # csv_logger_params = dict(save_dir="./experiments", version=args.save_dir.split("/")[-1])
    #     csv_logger_params = dict(
    #         save_dir="./experiments",
    #         name=os.path.join(*args.save_dir.split("/")[1:-1]),
    #         version=args.save_dir.split("/")[-1],
    #     )
    # # print(args.save_dir.split('/'))
    # # return
    # # if args.model_type == "stages":
    # #     wandb_logger_params.update(dict(name=args.model_name))
    # #     csv_logger_params.update(dict(name=args.model_name))
    # # elif args.model_type in ["massc", "massc_v2", "simple_massc", "avgloss_massc", "utime"]:
    # # wandb_logger_params.update(dict(name=args.model_type))
    # try:
    #     wandb_logger_params.update(dict(name=os.path.join(*args.save_dir.parts[1:])))
    # except AttributeError:
    #     wandb_logger_params.update(dict(name=os.path.join(*args.save_dir.split("/")[1:])))
    # checkpoint_monitor = pl_callbacks.ModelCheckpoint(
    #     filepath=os.path.join(args.save_dir, "{epoch:03d}-{eval_loss:.2f}"),
    #     monitor="eval_loss",
    #     save_last=True,
    #     save_top_k=1,
    # )
    # # csv_logger_params.update(dict(name=args.save_dir.split("/")[1]))
    # # tb_logger = pl_loggers.TensorBoardLogger("logs/")
    # # csv_logger_params.update(dict(name=args.save_dir.parts[1]))

    # # lr_monitor = pl_callbacks.LearningRateMonitor()
    # csv_logger = pl_loggers.CSVLogger(**csv_logger_params)
    # if args.debug:
    #     wandb_logger_params.update(dict(offline=True))
    # wandb_logger = pl_loggers.WandbLogger(**wandb_logger_params)
    # wandb_logger.watch(model)
    # # if not args.debug:
    # #     wandb_logger = WandbLogger(**wandb_logger_params)
    # #     wandb_logger.watch(model)
    # # else:
    # #     wandb_logger = None
    # # wandb_logger = None
    # loggers = [
    #     csv_logger,
    #     # tb_logger,
    #     wandb_logger,
    # ]
    # callbacks = [
    #     pl_callbacks.EarlyStopping(monitor="eval_loss", patience=args.earlystopping_patience),
    #     pl_callbacks.LearningRateMonitor(),
    # ]

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=loggers, callbacks=callbacks)

    # ================================================================================================================
    # LEARNING RATE FINDER ROUTINE
    # ----------------------------------------------------------------------------------------------------------------
    if args.lr_finder:
        lr_finder = trainer.tuner.lr_find(model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("results/lr_finder/test.png")
        return
    # ================================================================================================================

    # Fit model using trainer
    trainer.fit(model, dm)

    return 0

    # Return results on eval data
    predictions = trainer.test(
        # model,
        test_dataloaders=dm.val_dataloader(),
        # ckpt_path="best",
        # ckpt_path=trainer.checkpoint_callback.best_model_path,
        verbose=False,
    )
    if not model.use_ddp or (model.use_ddp and torch.distributed.get_rank() == 0):
        predictions = predictions[0]
        with open(os.path.join(args.save_dir, f"eval_predictions.pkl"), "wb") as pkl:
            pickle.dump(predictions, pkl)

        # eval_windows = [1, 15, 30]
        # eval_windows = [1, 2]  # STAGES predicts every 15 s
        eval_windows = [1]  # MASSC predicts every 30 s for evaluation
        df, cm_sub, cm_tot = utils.evaluate_performance(predictions, evaluation_windows=eval_windows, cases=["all"])
        best_acc = []
        best_kappa = []
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

                    best_acc.append(acc)
                    best_kappa.append(kappa)
        print(s)
        with open(os.path.join(args.save_dir, "eval_case_results.txt"), "w") as txt_file:
            print(s, file=txt_file)
        df.to_csv(os.path.join(args.save_dir, f"eval_results.csv"))
        with open(os.path.join(args.save_dir, f"eval_confusionmatrix.pkl"), "wb") as pkl:
            pickle.dump({"confusiomatrix_subject": cm_sub, "confusionmatrix_total": cm_tot}, pkl)

        try:
            trainer.logger.experiment.summary["best_acc"] = best_acc
            trainer.logger.experiment.summary["best_kappa"] = best_kappa
        except AttributeError:
            trainer.logger.experiment[1].summary["best_acc"] = best_acc
            trainer.logger.experiment[1].summary["best_kappa"] = best_kappa
    # # Run predictions on test data
    # if args.model_type == "stages":
    #     results_dir = os.path.join(
    #         "results",
    #         args.model_type,
    #         args.model_name,
    #         args.resume_from_checkpoint.split("/")[2],
    #         os.path.basename(wandb_logger.save_dir),
    #     )
    # elif args.model_type == "massc":
    #     results_dir = Path(os.path.join(args.save_dir, "results"))
    #     results_dir.mkdir(parents=True, exist_ok=True)
    #     # results_dir = os.path.join(
    #     #     "results", args.model_type, args.resume_from_checkpoint.split("/")[2], os.path.basename(wandb_logger.save_dir),
    #     # )
    # # if not os.path.exists(results_dir):
    # #     os.makedirs(results_dir)
    # # test_data = DataLoader(SscWscPsgDataset("./data/test/raw/ssc_wsc"), num_workers=args.n_workers, pin_memory=True)
    # # results = trainer.test(test_dataloaders=test_data, verbose=False)[0]
    # # evaluate_performance(results)
    # # print(len(results.keys()))
    # # with open(os.path.join(results_dir, 'SSC_WSC.pkl'), 'wb') as pkl:
    # #     pickle.dump(results, pkl)

    # # KHC data
    # khc_data = DataLoader(KoreanDataset(), num_workers=args.n_workers, pin_memory=True)
    # results = trainer.test(test_dataloaders=khc_data, verbose=False)
    # df = evaluate_performance(results)
    # print(len(results.keys()))
    # with open(os.path.join(results_dir, "KHC.pkl"), "wb") as pkl:
    #     pickle.dump(results, pkl)

    # df.to_csv(os.path.join(results_dir, 'KHC.csv'))

    # results = trainer.test(verbose=False)
    # test_params = dict(num_workers=args.n_workers, pin_memory=True)

    # test_data = DataLoader(datasets.SscWscPsgDataset(data_dir=args.data_dir, overlap=False, n_records=20, scaling="robust"))
    # test_data = DataLoader(SscWscPsgDataset("./data/test/raw/ssc_wsc", overlap=False, n_records=10), **test_params)
    # run_testing(test_data, "SSC-WSC")

    return 0


if __name__ == "__main__":
    run_training()

    # parser = ArgumentParser(add_help=False)

    # # add args from trainer
    # parser = Trainer.add_argparse_args(parser)

    # # Check the supplied model type
    # parser.add_argument("--model_type", type=str, default="massc")
    # temp_args, _ = parser.parse_known_args()

    # # give the module a chance to add own params
    # # good practice to define LightningModule speficic params in the module
    # if temp_args.model_type == "stages":
    #     parser = StagesModel.add_model_specific_args(parser)
    # elif temp_args.model_type == "massc":
    #     parser = MasscModel.add_model_specific_args(parser)
    # elif temp_args.model_type == "simple_massc":
    #     parser = SimpleMasscModel.add_model_specific_args(parser)
    # elif temp_args.model_type == "avgloss_massc":
    #     parser = AvgLossMasscModel.add_model_specific_args(parser)
    # elif temp_args.model_type == "utime":
    #     parser = UTimeModel.add_model_specific_args(parser)

    # # parse params
    # args = parser.parse_args()

    # args = utils.get_args()

    # if args.model_type in ["massc", "simple_massc", "avgloss_massc"]:
    #     args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
    # elif args.model_type == "utime":
    #     args.save_dir = Path(os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S")))
    #     # args.save_dir.mkdir(parents=True, exist_ok=True)

    # main(args)
