from argparse import ArgumentParser
from pathlib import Path

import torch

from sleep_staging.utils.model_utils import get_model_from_ckpt


def get_model_summary(model_path: str, output_path: str) -> None:

    model = get_model_from_ckpt(ckpt_path=model_path, device=torch.device('cpu'))
    print(f'Model architecture:', file=open(output_path, 'w'))
    print(model, file=open(output_path, 'a'))
    print(f'Model input/output:', file=open(output_path, 'a'))
    print(model.summarize('full'), file=open(output_path, 'a'))
    total_params = sum([p.nelement() for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f'\nTotal number of trainable parameters in model: {total_params:,}', file=open(output_path, 'a'))
    print(f'\nModel hyperparameters:\n{model.hparams}', file=open(output_path, 'a'))

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('--model-path', required=True, type=str, help='Path to model file (.ckpt)')
    parser.add_argument('--output-path', default='model_summary.txt', type=str, help='Path to output file containing model structure.')
    args = parser.parse_args()

    get_model_summary(args.model_path, args.output_path)