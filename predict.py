import os
import pprint
import pickle
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

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

    args = utils.get_args()

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
    # if args.model_type == "stages":
    #     # Update some hyperparameters
    #     # if args.model_name.split("_")[1] == "rh":
    #     #     args.n_hidden_units = np.random.randint(256, 256 + 128)  # np.random.randint(0.5 * 256, 1.5 * 256 + 1)
    #     model = StagesModel()
    # elif args.model_type == "massc":
    #     model = MasscModel()
    # elif args.model_type == "massc_v2":
    #     model = MasscV2Model.load_from_checkpoint(args.resume_from_checkpoint)
    # elif args.model_type == "simple_massc":
    #     model = SimpleMasscModel(**vars(args))
    # elif args.model_type == "avgloss_massc":
    #     model = AvgLossMasscModel(**vars(args))
    # elif args.model_type == "utime":
    #     model = UTimeModel(**vars(args))
    # else:
    #     raise NotImplementedError

    # Setup callbacks
    # wandb_logger_params = dict(save_dir="experiments", project="sleep-staging", log_model=True)
    # wandb_logger_params = dict(project="sleep-staging", log_model=False)
    # if args.model_type == "stages":
    # wandb_logger_params.update(dict(name=args.model_name))
    # elif args.model_type == "massc":
    # wandb_logger_params.update(dict(name=args.model_type))
    # wandb_logger_params.update(dict(name=os.path.join(*args.save_dir.parts[1:]), id=datetime.now().strftime("%Y%m%d_%H%M%S")))
    # checkpoint_logger = ModelCheckpoint(filepath=os.path.join(args.save_dir, "{epoch}"))
    # lr_logger = LearningRateLogger()
    # wandb_logger = WandbLogger(**wandb_logger_params)
    # wandb_logger
    # wandb_logger.watch(model)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, logger=False, deterministic=True)

    # # Fit model using trainer
    # trainer.fit(model)

    # ------------------------------------------------------------------------------- #
    # TEST ON NEW DATA
    # ------------------------------------------------------------------------------- #
    results_dir = os.path.dirname(args.resume_from_checkpoint)
    test_params = dict(num_workers=args.n_workers, pin_memory=True, shuffle=False)

    test_dm = []
    ds_args = model.hparams
    ds_args["n_jobs"] = args.n_jobs
    ds_args["n_records"] = args.n_records
    ds_args["n_workers"] = args.n_workers
    ds_args["limit_test_batches"] = args.limit_test_batches
    test_dm.append(("SSC-WSC", datasets.SscWscDataModule(**ds_args)),)
    for dm in test_dm:
        dm[1].setup("test")
    # model.dataset_params["n_records"] = 5

    # test_datasets.append(("SSC-WSC", datasets.SscWscPsgDataset(**model.dataset_params)))
    # model.dataset_params["n_records"] = -1
    # model.dataset_params["overlap"] = False
    # model.dataset_params["data_dir"] = None
    # test_datasets.append(("KHC", datasets.KoreanDataset(**model.dataset_params)))
    # test_datasets = [
    #     # ("SSC-WSC", datasets.SscWscPsgDataset(**model.dataset_params)),
    #     ("KHC", datasets.KoreanDataset(**model.dataset_params)),
    #     # ("KHC", datasets.KoreanDataset(overlap=False, scaling="robust", adjustment=30)),
    # ]
    # test_dataloaders = [(td[0], DataLoader(td[1], **test_params)) for td in test_datasets]

    # for name, tdl in test_dataloaders:
    for name, tdm in test_dm:
        # predictions = trainer.test(model, test_dataloaders=tdl, verbose=False)[0]
        predictions = trainer.test(model, datamodule=tdm, verbose=False)
        # trainer.test(model, datamodule=tdm, verbose=False)
        if not model.use_ddp or (model.use_ddp and torch.distributed.get_rank() == 0):
            # with open(os.path.join(results_dir, f"{name}_predictions.pkl"), "rb") as pkl:
            #     predictions = pickle.load(pkl)
            predictions = predictions[0]
            with open(os.path.join(results_dir, f"{name}_predictions.pkl"), "wb") as pkl:
                pickle.dump(predictions, pkl)

            df, cm_sub, cm_tot = utils.evaluate_performance(predictions, evaluation_windows=[1, 15, 30], cases=["all"])
            with np.printoptions(precision=3, suppress=True):
                for eval_window in cm_tot.keys():
                    print()
                    print(f"Evaluation window - {eval_window}")
                    print(f"---------------------------------")
                    for case in cm_tot[eval_window].keys():
                        print(f"Case: {case}")
                        print(cm_tot[eval_window][case])
                        NP = cm_tot[eval_window][case].sum(axis=1)
                        PP = cm_tot[eval_window][case].sum(axis=0)
                        precision = np.diag(cm_tot[eval_window][case]) / PP
                        recall = np.diag(cm_tot[eval_window][case]) / NP
                        f1 = 2 * precision * recall / (precision + recall)
                        print(f"F1: {f1}")
                        print()
            # pprint.pprint(cm_tot[1])
            # pprint.pprint(cm_tot[30])
            df.to_csv(os.path.join(results_dir, f"{name}_results.csv"))
            with open(os.path.join(results_dir, f"{name}_confusionmatrix.pkl"), "wb") as pkl:
                pickle.dump({"confusiomatrix_subject": cm_sub, "confusiomatrix_total": cm_tot}, pkl)

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
