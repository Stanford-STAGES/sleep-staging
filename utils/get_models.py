import numpy as np
import torch

import models


def get_model(args):

    if args.model_type == "stages":
        # TODO (alexno Sep 29, 2020): fix this
        # Update some hyperparameters
        if args.model_name.split("_")[1] == "rh":
            args.n_hidden_units = np.random.randint(256, 256 + 128)  # np.random.randint(0.5 * 256, 1.5 * 256 + 1)
        model = models.StagesModel(**vars(args))
    elif args.model_type == "massc":
        if args.resume_from_checkpoint:
            # model = models.MasscModel.load_from_checkpoint(args.resume_from_checkpoint)
            args = torch.load(args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"]
            model = models.MasscModel(args)
        else:
            # model = models.MasscModel(**vars(args))
            model = models.MasscModel(vars(args))
    elif args.model_type == "massc_v2":
        model = models.MasscV2Model.load_from_checkpoint(args.resume_from_checkpoint)
    elif args.model_type == "simple_massc":
        model = models.SimpleMasscModel(**vars(args))
    elif args.model_type == "avgloss_massc":
        model = models.AvgLossMasscModel(**vars(args))
    elif args.model_type == "utime":
        # model = models.UTimeModel(**vars(args))
        model = models.UTimeModel(vars(args))
    else:
        raise NotImplementedError

    return model
