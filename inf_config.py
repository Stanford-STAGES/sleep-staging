import os
import numpy as np


# Required code indentations
class Config(object):

    @staticmethod
    def get(model_name):
        if model_name[0:2] == 'ac':
            return ACConfig(model_name)
        else:
            raise Exception

    def __getitem__(self, itemname):
        return object.__getattribute__(self, itemname)

    def __init__(self, scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size, max_train_len, atonce,
                 restart=True, model_name='small_lstm', is_train=False):
        # root_python = os.path.dirname(os.path.realpath(__file__))
        # root_base = str.join('/', root_python.split('/')[:-1])
        # Model folder - relative to this location
        root_model_dir = '../';  # Change this if models are saved elsewhere

        self.model_dir = os.path.join(root_model_dir, 'model', scope, model_name)

        # Data
        # data_dir = '';
        self.train_data = os.path.join('/media/neergaard/Storage/jens/', 'ac_training_data')
        # self.test_data = os.path.join('/media/neergaard/neergaardhd/jens', 'ac_testing_data/ac_data_test_new')
        # self.test_data = os.path.join('/media/neergaard/neergaardhd/REPLICATION_DATA', 'CHINESE_DATA/H5')
        self.test_data = os.path.join('/media/neergaard/neergaardhd/REPLICATION_DATA', 'FRENCH_DATA', 'H5')

        # Configuration
        self.model_name = model_name
        self.scope = scope
        self.is_training = is_train
        self.num_features = num_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.restart = restart
        self.lstm = lstm
        self.num_hidden = num_hidden
        self.keep_prob = 0.5
        self.segsize = segsize
        self.eval_nseg_atonce = atonce
        self.max_train_len = max_train_len
        self.save_freq = 2
        self.max_steps = 2000
        self.init_learning_rate = 0.005
        self.learning_rate_decay = 750

    def checkpoint_file(self, ckpt=0):
        if ckpt == 0:
            return os.path.join(self.model_dir, 'model.ckpt')
        else:
            return os.path.join(self.model_dir, 'model.ckpt-%.0f' % ckpt)


class ACConfig(Config):

    def __init__(self, restart=True, model_name='ac_rh_ls_lstm_01', is_training=False):
        print('model: ' + model_name)
        if model_name[3:5] == 'lh':
            num_hidden = 256
        elif model_name[3:5] == 'rh':
            np.random.seed(int(model_name[-2:]))
            num_hidden = 256 + np.round(np.random.rand(1) * 128)
            num_hidden = num_hidden[0].astype(int)
        else:
            num_hidden = 128

        if model_name[6:8] == 'ls':
            segsize = 60
            atonce = 1000
        else:
            segsize = 20
            atonce = 3000

        if model_name[9:11] == 'ff':
            lstm = False
        else:
            lstm = True

        if is_training:
            batch_size = 30
        else:
            batch_size = 1

        scope = 'ac'
        num_features = 1640
        num_classes = 5
        max_train_len = 14400
        super(ACConfig, self).__init__(scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size,
                                       max_train_len, atonce, restart, model_name, is_training)
