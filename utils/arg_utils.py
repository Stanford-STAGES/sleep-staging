import argparse
import json
import os
import pprint
from datetime import datetime
from glob import glob

import torch
import pytorch_lightning as pl

import datasets
import models


def get_args(stage="train", print_args=False):

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint_monitor", default="eval_loss", type=str)
    parser.add_argument("--dataset_type", type=str, default="ssc-wsc")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--earlystopping_monitor", default="eval_loss", type=str)
    parser.add_argument("--earlystopping_patience", default=100, type=int)
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--lr_finder", action="store_true")
    parser.add_argument("--seed", default=1337, type=int)
    if stage == "predict":
        parser.add_argument("--predict_on", type=json.loads)

    # add args from trainer
    parser = pl.Trainer.add_argparse_args(parser)

    # Check the supplied model type
    parser.add_argument("--model_type", type=str, default="massc")
    temp_args, _ = parser.parse_known_args()

    # Optionally resume from checkpoint
    if temp_args.resume_from_checkpoint and os.path.isdir(temp_args.resume_from_checkpoint):
        temp_args.resume_from_checkpoint = glob(os.path.join(temp_args.resume_from_checkpoint, "epoch*.ckpt"))[0]
    if temp_args.resume_from_checkpoint:
        hparams = torch.load(temp_args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
        temp_args.model_type = hparams["model_type"]

    # add args from dataset
    try:
        parser = datasets.available_datamodules[temp_args.model_type].add_dataset_specific_args(parser)
    except:
        parser = datasets.available_datamodules["ssc-wsc"].add_dataset_specific_args(parser)

    # Add args from model
    parser = models.available_models[temp_args.model_type].add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    # Check the sequence length argument
    if isinstance(args.sequence_length, str) and args.sequence_length != "full":
        try:
            args.sequence_length = int(args.sequence_length)
        except:
            print(f"args.sequence_length={args.sequence_length} can't be converted to int, set to default value.")
            args.sequence_length = 5

    # update args from hparams
    if args.resume_from_checkpoint:
        args.model_type = hparams["model_type"]

    # Create a save directory
    if not args.resume_from_checkpoint:
        # if args.model_type in ["massc", "simple_massc", "avgloss_massc", "massc_v2"]:
        if "massc" in args.model_type:
            # args.save_dir = Path(os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S")))
            if args.name is None:
                # args.save_dir = os.path.join("experiments", "massc", datetime.now().strftime("%Y%m%d_%H%M%S"))
                args.save_dir = os.path.join("experiments", "massc")
            else:
                args.save_dir = os.path.join("experiments", "massc", args.name,)
            #     args.save_dir = os.path.join("experiments", "massc", args.name, datetime.now().strftime("%Y%m%d_%H%M%S"),)
        elif args.model_type == "utime":
            # args.save_dir = Path(os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S")))
            # args.save_dir = os.path.join("experiments", "utime", datetime.now().strftime("%Y%m%d_%H%M%S"))
            args.save_dir = os.path.join("experiments", "utime")
        elif args.model_type == "stages":
            if args.name is None:
                args.save_dir = os.path.join("experiments", args.model_type, args.model_name,)
            else:
                args.save_dir = os.path.join("experiments", args.model_type, args.model_name, args.name,)
        else:
            args.save_dir = os.path.join("experiments", args.model_type, datetime.now().strftime("%Y%m%d_%H%M%S"))
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            os.makedirs(args.save_dir, exist_ok=True)
    else:
        args.save_dir = (
            args.resume_from_checkpoint
            if os.path.isdir(args.resume_from_checkpoint)
            else os.path.dirname(args.resume_from_checkpoint)
        )
    # os.makedirs(args.save_dir, exist_ok=True)
    # args.save_dir.mkdir(parents=True, exist_ok=True)

    # # Check if we're resuming from checkpoint
    # if args.resume_from_checkpoint:
    #     print("hej")

    # Get the best model from the directory by default
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        args.resume_from_checkpoint = temp_args.resume_from_checkpoint
        # args.resume_from_checkpoint = glob(os.path.join(args.resume_from_checkpoint, "epoch*.ckpt"))[0]

    # Set the seed
    if args.model_type == "stages":
        args.seed = 42 + int(args.model_name.split("_")[-1])
    else:
        args.seed = 1337

    # # Override n_workers for prediction
    # if stage == "predict":
    #     if temp_args.n_workers is not None:
    #         args.n_workers = temp_args.n_workers

    # Iff sequence_length is full, we can only accept batch_size 1
    if isinstance(args.sequence_length, str):
        assert args.batch_size == 1, f"Can only use batch_size==1 when sequence_length=='full'"

    if print_args:
        pprint.pprint(vars(args), indent=4)

    return args
