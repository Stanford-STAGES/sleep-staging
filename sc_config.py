import os
import numpy as np

class Config(object):

    @staticmethod
    def get(model_name):
        if model_name[0:3] == 'oct':
            return OCConfig(model_name)
        elif model_name[0:2] == 'ac':
            return ACConfig(model_name)
        else:
            raise Exception

    def __init__(self, scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size, max_train_len, atonce, restart=True, model_name='small_lstm', is_train=False):

        # Model folder
        root_python = os.path.dirname(os.path.realpath(__file__))
        root_base = str.join('/', root_python.split('/')[:-1])
        
        self.model_dir = os.path.join(root_python, 'model', scope, model_name)
        self.model_dir_test = os.path.join(root_python, 'model', scope, model_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        # Data
        data_dir = '/scratch/users/jenss/'
        
	if model_name[0]=='o':
	        self.train_data = os.path.join(data_dir, 'octave_training_data')
		self.test_data = os.path.join(data_dir, 'octave_test_data2')

        else:
		self.train_data = os.path.join(data_dir, 'ac_training_data')
		self.test_data = os.path.join(data_dir, 'ac_test_data')

        self.train_dir = os.path.join(self.model_dir, 'train')
        if not os.path.isdir(self.train_dir):
            os.mkdir(self.train_dir)

        # Configuration
        self.model_name = model_name
        self.scope = scope
        self.load_list = self.model_dir+'_load_list.csv'
        self.load_prob_file = self.model_dir+'_load_list.h5'
        self.validation_results = self.model_dir+'_training_data.csv'
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
	self.save_freq = 1

        # Training
        self.max_steps = 500000

    def checkpoint_file(self, ckpt=0):
        if ckpt == 0:
            return os.path.join(self.model_dir, 'model.ckpt')
        else:
            return os.path.join(self.model_dir, 'model.ckpt-%.0f' % ckpt)


class OCConfig(Config):

    def __init__(self,restart=True, model_name='oct_sh_ss_lstm', is_training=False):
        print('model: '+model_name)
        scope = 'oct'
        num_features = 25
        num_classes = 5

        max_train_len = 360000


        if is_training:
            batch_size = 15
        else:
            batch_size = 1
        
        if model_name[4:6]=='lh':
            num_hidden = 256
        else:
            num_hidden = 128
        if model_name[7:9]=='ls':
            segsize = 1500
	    atonce = 1500
        else:
            segsize = 500
	    atonce = 4500
        if model_name[10:12]=='ff':
            lstm = False
        else:
            lstm = True

        
        super(OCConfig, self).__init__(scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size, max_train_len, atonce, restart, model_name, is_training)

class ACConfig(Config):
    
    def __init__(self,restart=True, model_name='ac_sh_ss_lstm', is_training=False):
        print('model: '+model_name)
        scope = 'ac'
        num_features = 1640
        num_classes = 5
	if is_training:
        	batch_size = 5
	else:
		batch_size = 1

        max_train_len = 14400
        
        if model_name[3:5]=='lh':
            num_hidden = 256
	elif model_name[3:5]=='rh':
	    	np.random.seed(int(model_name[-2:]))
	    	num_hidden = 256+np.round(np.random.rand(1)*128)
		num_hidden = num_hidden[0].astype(int)
        else:
            num_hidden = 128
        if model_name[6:8]=='ls':
            segsize = 60
	    atonce = 700
        else:
            segsize = 20
	    atonce = 3000
        if model_name[9:11]=='ff':
            lstm = False
        else:
            lstm = True

        
        super(ACConfig, self).__init__(scope, num_features, num_hidden, segsize, lstm, num_classes, batch_size, max_train_len, atonce, restart, model_name, is_training)
