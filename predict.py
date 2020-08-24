import os
import pickle
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler
from torch.utils.data import DataLoader

from datasets import *
from models import *
from utils import *


def main(args):

    # Remember to seed!
    if args.model_type == "stages":
        seed = int(args.model_name.split("_")[-1])
    else:
        seed = 1337
    seed_everything(seed)

    # Setup model
    if args.model_type == "stages":
        # Update some hyperparameters
        # if args.model_name.split("_")[1] == "rh":
        #     args.n_hidden_units = np.random.randint(256, 256 + 128)  # np.random.randint(0.5 * 256, 1.5 * 256 + 1)
        model = StagesModel()
    elif args.model_type == "massc":
        model = MasscModel()
    elif args.model_type == "massc_v2":
        model = MasscV2Model.load_from_checkpoint(args.resume_from_checkpoint)
    else:
        raise NotImplementedError

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
    test_params = dict(num_workers=args.n_workers, pin_memory=True)

    def run_testing(_data, _name):
        predictions = trainer.test(model, test_dataloaders=_data, verbose=False)[0]
        with open(os.path.join(results_dir, f"{_name}_predictions.pkl"), "wb") as pkl:
            pickle.dump(predictions, pkl)

        df_results = evaluate_performance(predictions)
        df_results.to_csv(os.path.join(results_dir, f"{_name}_results.csv"))

    # Run predictions on SSC-WSC test data
    test_data = DataLoader(SscWscPsgDataset("./data/test/raw/ssc_wsc"), **test_params)
    run_testing(test_data, "SSC-WSC")

    # Test on KHC data
    khc_data = DataLoader(KoreanDataset(), **test_params)
    run_testing(khc_data, "KHC")

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

    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # Check the supplied model type
    parser.add_argument("--model_type", type=str, default="massc")
    temp_args, _ = parser.parse_known_args()

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    if temp_args.model_type == "stages":
        parser = StagesModel.add_model_specific_args(parser)
    elif temp_args.model_type == "massc":
        parser = MasscModel.add_model_specific_args(parser)
    elif temp_args.model_type == "massc_v2":
        parser = MasscV2Model.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    # if args.model_type in ["massc", "massc_v2"]:
    # args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
    # args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
