import os
import pickle
from argparse import ArgumentParser
from datetime import datetime

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
    wandb_logger
    wandb_logger.watch(model)

    # Define trainer object from arguments
    trainer = Trainer.from_argparse_args(args, deterministic=True, logger=wandb_logger, callbacks=[lr_logger])

    # Fit model using trainer
    trainer.fit(model)

    # Run predictions on test data
    if args.model_type == 'stages':
        results_dir = os.path.join('results', args.model_type, args.model_name, args.resume_from_checkpoint.split('/')[2], os.path.basename(wandb_logger.save_dir))
    elif args.model_type == 'massc':
        results_dir = os.path.join('results', args.model_type, args.resume_from_checkpoint.split('/')[2], os.path.basename(wandb_logger.save_dir))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    test_data = DataLoader(SscWscPsgDataset('./data/test/raw/individual_encodings'), pin_memory=True)
    results = trainer.test(test_dataloaders=test_data, verbose=False)[0]
    print(len(results.keys()))
    with open(os.path.join(results_dir, 'SSC_WSC.pkl'), 'wb') as pkl:
        pickle.dump(results, pkl)
    return 0

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
