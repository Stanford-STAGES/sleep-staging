from argparse import Namespace

import torch

import sleep_staging.models as models


def get_model(args):

    if args.resume_from_checkpoint:
        resume_from_checkpoint = args.resume_from_checkpoint
        args = Namespace(
            **(torch.load(args.resume_from_checkpoint, map_location=torch.device("cpu"))["hyper_parameters"])
        )
        args.resume_from_checkpoint = resume_from_checkpoint
        # model = models.available_models[args.model_type].load_from_checkpoint(args.resume_from_checkpoint)
    # else:
    try:
        model = models.available_models[args.model_type](**vars(args))
    except TypeError:
        model = models.available_models[args.model_type](vars(args))

    return model


def get_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_type = ckpt["hyper_parameters"]["model_type"]
    return models.available_models[model_type].load_from_checkpoint(ckpt_path, map_location="cpu").eval().to(device)
