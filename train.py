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
from pytorch_lightning.loggers import CSVLogger
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
        if args.model_name.split("_")[1] == "rh":
            args.n_hidden_units = np.random.randint(256, 256 + 128)  # np.random.randint(0.5 * 256, 1.5 * 256 + 1)
        model = StagesModel(**vars(args))
    elif args.model_type == "massc":
        model = MasscModel(**vars(args))
    elif args.model_type == "massc_v2":
        model = MasscV2Model(**vars(args))
    else:
        raise NotImplementedError

    # Setup callbacks
    # wandb_logger_params = dict(save_dir="experiments", project="sleep-staging", log_model=True)
    wandb_logger_params = dict(project="sleep-staging", log_model=False)
    csv_logger_params = dict(save_dir="./experiments", version=args.save_dir.parts[-1])
    if args.model_type == "stages":
        wandb_logger_params.update(dict(name=args.model_name))
    elif args.model_type in ["massc", "massc_v2"]:
        # wandb_logger_params.update(dict(name=args.model_type))
        wandb_logger_params.update(dict(name=os.path.join(*args.save_dir.parts[1:])))
        checkpoint_logger = ModelCheckpoint(filepath=os.path.join(args.save_dir, "{epoch:03d}_best"), save_last=True)
        csv_logger_params.update(dict(name=args.save_dir.parts[1]))
    lr_logger = LearningRateLogger()
    csv_logger = CSVLogger(**csv_logger_params)
    wandb_logger = WandbLogger(**wandb_logger_params)
    wandb_logger.watch(model)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(
        args, deterministic=True, checkpoint_callback=checkpoint_logger, logger=[csv_logger, wandb_logger], callbacks=[lr_logger]
    )

    # Fit model using trainer
    trainer.fit(model)

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

    if args.model_type in ["massc", "massc_v2"]:
        args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
        # args.save_dir.mkdir(parents=True, exist_ok=True)

    main(args)
