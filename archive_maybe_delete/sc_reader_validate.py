import numpy as np
import collections
import h5py
import csv
import os

import tensorflow as tf
#from tensorflow.python.platform import gfile

import sc_config

Dataset = collections.namedtuple('Dataset', ['data', 'target'])


class ScoreData:

    def __init__(self, pathname, config):
        self.pathname = pathname
        self.features = []
        self.logits = []
        self.num_batches = 0
        self.iter_batch = -1
        self.iter_rewind = 0
        self.config = config
        self.batch_order = np.array([])
        self.batch_accuracy = np.array([])
        self.batch_cost = np.array([])
        self.batch_cross_ent = np.array([])
        self.anyleft = True
        self.valid_list = []
        self.losses = []
        self.loss_temp = []
        self.cross_ents = []
        self.cross_temp = []
        self.accuracies = []
        self.acc_temp = []
        self.baselines = []
        self.base_temp = []
        self.is_long = False
        self.load_list()
        #self.load()

    def __iter__(self):

        return self

    def next(self):
        #Load data
        if self.iter_batch == self.num_batches:    
            self.load()
            
        # Increment counters
        self.iter_batch += 1

        # Determine stopping criteria
        if (self.iter_batch + 1) > len(self.batch_order):
            raise StopIteration()

        # Return relevant batch
        x, y = self.get_batch(self.iter_batch)
        return x, y



    def get_batch(self, batch_num):

        # Find indices
        batch_num_ordered = self.batch_order[batch_num]
        if self.is_long:
            self.indL = np.arange(batch_num*self.config.eval_nseg_atonce, np.min([(batch_num+1)*self.config.eval_nseg_atonce,self.logits.shape[1]]),
                                  step=1,
                                  dtype=np.int)
	    print('Label index')
            print(self.indL.shape)
            indD = np.arange(batch_num*self.config.eval_nseg_atonce*self.config.segsize, np.min([(batch_num+1)*self.config.eval_nseg_atonce*self.config.segsize,self.features.shape[1]]),
                             step=1,
                             dtype=np.int)
	    print(indD.shape)
            # Find batches
            x = self.features[:,indD, :]
            t = self.logits[:,self.indL,:]
        else:
                
            ind = np.arange(batch_num_ordered*self.batch_size, (batch_num_ordered+1)*self.batch_size,
                        step=1,
                        dtype=np.int)

        # Find batches
            x = self.features[ind, :, :]
            t = self.logits[ind,:,:]
            t = np.reshape(t,[-1,self.config.num_classes])

        # Return
        return x, t
    def load_list(self):
        
	with open(os.getcwd()+'/'+self.config.model_name + 'validationlist.csv','rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                self.valid_list += row
        
	print('List of files.. ')
	print(self.valid_list)
    def load(self):
        if len(self.loss_temp)!=0:
	    self.summarize_results()
        # Import from CSV file
	print(self.valid_list)
        if len(self.valid_list)>0:
            self.filename = self.valid_list.pop(0)
        else:
           
            self.save_results()
            self.anyleft = False            
	    print('Stopping..')
            raise StopIteration()
	   
        data_set = self.load_h5()
                

        self.features = data_set.data
 
               
        assert np.round(self.num_batches, 0) == self.num_batches

        # Build logits from labels (labels can be interpreted as p(arousal)
        labels = np.transpose(data_set.target, axes=[1, 2, 0])    
        self.n_seg = labels.shape[0]//self.config.segsize
	print(labels.shape)
	print(self.config.segsize)
	print(self.n_seg)
	print(self.config.num_classes)
        labels = np.reshape(labels,[self.config.segsize, self.n_seg,self.config.num_classes,labels.shape[2]],order='F')
        self.logits = np.transpose(np.mean(labels,axis=0),axes=[2, 0, 1])
        self.batch_order = np.arange(self.num_batches)

        # Check that logits are all good
        #assert np.all(np.sum(self.logits, 1), 0)
        self.batch_accuracy = np.full(self.num_batches, 0, np.float32)
        self.batch_cost = np.full(self.num_batches, 0, np.float32)
        self.batch_cross_ent = np.full(self.num_batches, 0, np.float32)
        self.iter_batch = -1

    def load_h5(self):

        # Read from file

        #with gfile.Open(self.filename) as h5_file:
        f = h5py.File(self.filename,'r')
    
            
        dataT = f[u'/valD']
        targetT = f[u'/valL']
	print(np.rank(dataT))
	if np.rank(dataT)==2:
		print(dataT.shape)
        	self.is_long = True
        	self.n_seg = dataT.shape[0]//self.config.segsize 
		dataT = dataT[:self.n_seg*self.config.segsize,:]
		targetT = targetT[:self.n_seg*self.config.segsize,:]
        	self.num_batches = np.floor(targetT.shape[0] / (self.config.eval_nseg_atonce*self.config.segsize))
	 	print(dataT.shape)
	 
        	self.batch_size = 1
        	dataT = np.expand_dims(dataT,0)
        	targetT = np.expand_dims(targetT,0)           
	else:
		self.is_long = False
        	self.batch_size = self.config.batch_size
        	self.num_batches = dataT.shape[0] // self.batch_size

	print(dataT.shape)
	print(targetT.shape)

        return Dataset(data=dataT, target=targetT)
    def record_results(self,loss,cross_ent,accuracy,baseline):

        
        self.loss_temp.insert(0,loss)
        self.acc_temp.insert(0,accuracy)
        self.base_temp.insert(0,baseline)
        self.cross_temp.insert(0,cross_ent)
        
    def summarize_results(self):    


        self.losses.insert(0,sum(self.loss_temp)/len(self.loss_temp))
        self.cross_ents.insert(0,sum(self.cross_temp)/len(self.cross_temp))
        self.accuracies.insert(0,sum(self.acc_temp)/len(self.acc_temp))
       
        self.loss_temp = []
        self.acc_temp = []
        self.cross_temp = []

    def save_results(self):
        print('Data being saved...')

        data = ['Val. Acc.','Val. Loss','Val. Cross-Ent']
	                        
        with open('Res_'+self.config.model_name + '_' + str(self.iter_rewind) + '_validationlist.csv', 'ab') as f:
            load_file_writer = csv.writer(f, delimiter=' ')
            load_file_writer.writerow(data)
            for i in range(len(self.losses)):
                data = [self.accuracies[i] , self.losses[i] , self.cross_ents[i]]
		
                load_file_writer.writerow(data)
