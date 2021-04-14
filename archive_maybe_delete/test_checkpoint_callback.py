import os
from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, idx, batch_idx):
        return 0

    def train_dataloader(self):
        return None

    def configure_optimizers(self):
        return None


def main(args):

    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(args.save_dir, "{epoch}"))
    trainer = Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    model = MyModel()
    trainer.fit(model)


if __name__ == "__main__":

    parser = ArgumentParser(add_help=False)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.save_dir = os.path.join(".", datetime.now().strftime("%Y%m%d_%H%M%S"))

    main(args)
