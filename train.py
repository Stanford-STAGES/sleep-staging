import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler

from models import *


def main(args):

    # Remember to seed!
    if args.model_type == 'stages':
        seed = int(args.model_name.split('_')[-1])
    else:
        seed = 1337
    seed_everything(seed)

    # Setup model
    if args.model_type == 'stages':
        # Update some hyperparameters
        if args.model_name.split('_')[1] == 'rh':
            args.n_hidden_units = np.random.randint(256, 256+128) # np.random.randint(0.5 * 256, 1.5 * 256 + 1)
        model = StagesModel(**vars(args))
    elif args.model_type == 'massc':
        model = MasscModel(**vars(args))
    else:
        raise NotImplementedError

    # Setup callbacks
    wandb_logger_params = dict(save_dir='experiments', project='sleep-staging', log_model=True)
    if args.model_type == 'stages':
        wandb_logger_params.update(dict(name=args.model_name))
    elif args.model_type == 'massc':
        wandb_logger_params.update(dict(name=args.model_type))
    lr_logger = LearningRateLogger()
    wandb_logger = WandbLogger(**wandb_logger_params)
    wandb_logger.watch(model)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=wandb_logger, callbacks=[lr_logger])

    # Fit model using trainer
    trainer.fit(model)


if __name__ == '__main__':

    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # Check the supplied model type
    parser.add_argument('--model_type', type=str, default='massc')
    temp_args, _ = parser.parse_known_args()

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    if temp_args.model_type == 'stages':
        parser = StagesModel.add_model_specific_args(parser)
    elif temp_args.model_type == 'massc':
        parser = MasscModel.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
