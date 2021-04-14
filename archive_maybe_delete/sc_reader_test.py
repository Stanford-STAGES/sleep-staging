import numpy as np
import random
import time
import collections
import h5py
import csv
import os
import scipy.io as sio
import sc_config

import tensorflow as tf
Dataset = collections.namedtuple('Dataset', ['data', 'target'])


class ScoreData:


    def __init__(self, config):
        self.pathname = config.test_data
        self.filename = []
        self.features = []
        self.logits = []
        self.num_batches = 0
        self.subject_counter = -1
        self.iter_batch = -1
        self.config = config
        self.batch_order = np.array([])
        self.batch_accuracy = np.array([])
        self.batch_cost = np.array([])
        self.batch_cross_ent = np.array([])
        self.prediction = []
        self.accuracy = []
        self.indL = []
        self.predictions = []
	self.anyleft = True
	self.load()

    def __iter__(self):

        return self

    def next(self):
        #Load data
        if self.iter_batch==self.num_batches:
            self.load()
            self.iter_batch = -1
	    raise StopIteration()
        # Increment counters
        self.iter_batch += 1
	
        # Determine stopping criteria
        #if (self.iter_batch + 1) > self.num_batches:
        #    raise StopIteration()

        # Return relevant batch
        x, y = self.get_batch(self.iter_batch)
        return x, y

    def get_batch(self, batch_num):

        # Find indices
        self.indL = np.arange(batch_num*self.config.eval_nseg_atonce, np.min([(batch_num+1)*self.config.eval_nseg_atonce,self.logits.shape[0]]),
                              step=1,
                              dtype=np.int)
	print(self.indL.shape)
        indD = np.arange(batch_num*self.config.eval_nseg_atonce*self.config.segsize, np.min([(batch_num+1)*self.config.eval_nseg_atonce*self.config.segsize,self.features.shape[1]]),
                              step=1,
                              dtype=np.int)
        # Find batches
        x = self.features[:,indD, :]
        t = self.logits[self.indL,:]

        # Return
        return x, t

    def load(self):

	self.logits = []
	self.features = []

        if not self.filename:
            self.filename = os.listdir(self.pathname)
	    #self.filename.sort()
	    #self.filename = self.filename[3:]
	    random.seed(int(round(time.clock()*1000)))  
	    random.shuffle(self.filename)
	    print(self.filename)
	    print(len(self.filename))

        if (self.subject_counter+1)==len(self.filename):
	    self.anyleft = False
            raise StopIteration()
        self.subject_counter += 1

        data_set = self.load_h5()
	
        self.n_seg = data_set.target.shape[0]//self.config.segsize
	
	
        self.features = np.expand_dims(data_set.data[:self.n_seg*self.config.segsize,:],0)
        target = data_set.target[:self.n_seg*self.config.segsize,:]

	self.num_batches = np.ceil(np.divide(target.shape[0],(self.config.eval_nseg_atonce*self.config.segsize),dtype='float'))

	self.Nextra = np.ceil(self.num_batches * self.config.eval_nseg_atonce * self.config.segsize)%target.shape[0]
	
	meanF = np.mean(np.mean(self.features,0),0) * np.ones([1,self.Nextra,self.features.shape[2]])
	print(self.features.shape)
	print(meanF.shape)
	
	self.features = np.concatenate([self.features,meanF],1)
	print(self.features.shape)
	extraTarget = np.zeros([self.Nextra,5])
	extraTarget[:,0] = 1
	target = np.concatenate([target,extraTarget],0)

	self.n_seg = target.shape[0]//self.config.segsize
	
        target = np.reshape(target,[self.config.segsize, self.n_seg,self.config.num_classes],order='F')
	        
        self.logits = np.mean(target,axis=0)

	print('Total n batches '+str(self.num_batches))

    def load_h5(self):

        # Read from file

	savePath = os.path.join('/scratch/users/jenss/prediction_revisited','results')

        if not os.path.exists(savePath):
                os.makedirs(savePath)

	while True:
		
	        fileN = self.pathname+'/'+self.filename[self.subject_counter]
		self.saveName = 'results_' + self.config.model_name + '_' + self.filename[self.subject_counter] + '.mat'

	        self.saveName = os.path.join(savePath,self.saveName)
		if not(os.path.exists(self.saveName)):
			break
		else:
			print(self.saveName+' exists..')
			self.subject_counter += 1
			if self.subject_counter==len(self.filename)+1:
				raise StopIteration()
	print(self.filename[self.subject_counter])
        print('Loading '+fileN)
	print(fileN.find('.'))
	print(fileN)
        f = h5py.File(fileN,'r')
    
        data = f[u'/data']
        target = f[u'/labels']
	print(data)
	print(target)
        return Dataset(data=data, target=target)
        
    def record_results(self,prediction):
        if len(prediction)==0:
		self.save_results()
		return
        if not type(self.prediction).__module__=='numpy':
            self.prediction = np.zeros([self.logits.shape[0],self.logits.shape[1]])
	
        self.prediction[self.indL,:] = prediction
        
        if self.iter_batch==self.num_batches:
            self.save_results()
        
    def save_results(self):
        
	self.Nextra = self.Nextra/self.config.segsize

        print(self.saveName)

	print(self.Nextra)
	print(self.prediction.shape)

        saveDict = {'predictions': self.prediction[:-self.Nextra,:],
                    'targets': self.logits[:-self.Nextra,:]                       
                    }
        
        sio.savemat(self.saveName, saveDict)
	
	self.prediction = []
	self.logits = []

