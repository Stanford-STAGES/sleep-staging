import argparse
import os

import numpy as np


class Config(object):

    def __init__(self, parser):
        args = parser.parse_args()
        self.model_name = args.model
        if self.model_name[3:5] == 'lh':
            self.num_hidden = 256
        elif self.model_name[3:5] == 'rh':
            np.random.seed(int(self.model_name[-2:]))
            num_hidden = 256 + np.round(np.random.rand(1) * 128)
            self.num_hidden = num_hidden[0].astype(int)
        else:
            self.num_hidden = 128

        if self.model_name[6:8] == 'ls':
            self.segsize = 60
            self.atonce = 1000
        else:
            self.segsize = 20
            self.atonce = 3000

        if self.model_name[9:11] == 'ff':
            self.lstm = False
        else:
            self.lstm = True

        self.batch_size = 30
        self.scope = 'ac'
        self.num_features = 1640
        self.num_classes = 5
        self.max_train_len = 14400

        #


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Which model to build')
    config = Config(parser)
    print(config)
