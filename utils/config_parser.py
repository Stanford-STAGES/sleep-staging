import argparse
import os
from datetime import datetime

import torch
import pytorch_lightning as pl

import datasets
import models


def get_args():

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--earlystopping_patience", default=100, type=int)
    parser.add_argument("--lr_finder", action="store_true")

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # add args from dataset
    parser = datasets.SscWscDataModule.add_dataset_specific_args(parser)

    # Check the supplied model type
    parser.add_argument("--model_type", type=str, default="massc")
    temp_args, _ = parser.parse_known_args()

    if temp_args.resume_from_checkpoint:
        hparams = torch.load(temp_args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
        temp_args.model_type = hparams["model_type"]

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    if temp_args.model_type == "stages":
        parser = models.StagesModel.add_model_specific_args(parser)
    elif temp_args.model_type == "massc":
        parser = models.MasscModel.add_model_specific_args(parser)
    elif temp_args.model_type == "massc_v2":
        parser = models.MasscV2Model.add_model_specific_args(parser)
    elif temp_args.model_type == "simple_massc":
        parser = models.SimpleMasscModel.add_model_specific_args(parser)
    elif temp_args.model_type == "avgloss_massc":
        parser = models.AvgLossMasscModel.add_model_specific_args(parser)
    elif temp_args.model_type == "utime":
        parser = models.UTimeModel.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    # update args from hparams
    if temp_args.resume_from_checkpoint:
        args.model_type = hparams["model_type"]

    if args.model_type in ["massc", "simple_massc", "avgloss_massc"]:
        # args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
        if args.name is None:
            args.save_dir = os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S"))
        else:
            args.save_dir = os.path.join("experiments", "massc", args.name, datetime.now().strftime("%Y%m%d_%H%M%S"))
    elif args.model_type == "utime":
        # args.save_dir = Path(os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S")))
        args.save_dir = os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(args.save_dir, exist_ok=True)
    # args.save_dir.mkdir(parents=True, exist_ok=True)

    # # Check if we're resuming from checkpoint
    # if args.resume_from_checkpoint:
    #     print("hej")

    return args
