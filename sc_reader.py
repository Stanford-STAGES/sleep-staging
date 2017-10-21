import numpy as np
import collections
import h5py
import csv
import os
import random
import tempfile
#from tensorflow.python.platform import gfile

import sc_config
import tensorflow as tf
Dataset = collections.namedtuple('Dataset', ['data', 'target','weights'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class ScoreData:

    def __init__(self, pathname, config, num_steps=None):
        self.pathname = pathname
        self.filename = []
        self.features = []
        self.logits = []
        self.weights = []
        self.keep_these = []
        self.num_batches = 0
        self.num_steps = num_steps
        self.batch_size = 0
        self.seq_in_file = []
        self.iter_batch = -1
        self.iter_steps = -1
        self.iter_rewind = -1
        self.config = config
        self.batch_size
        self.batch_order = np.arange(1)
        self.batch_accuracy = np.array([])
        self.batch_cost = np.array([])
        self.confidence = np.array([])
        self.validation_files = []
        self.is_training = config.is_training
        self.should_save = False
	self.is_long = 0
        
        files = os.listdir(self.pathname)
	file = random.choice(files)
	
	if file[:4]=='long':
        	self.is_long = 3

	print(file[:4])
	print(self.is_long)
        self.filename = os.path.join(self.pathname,file)
	self.validation_files.insert(0,self.filename)

	self.load()
        
	

    def __iter__(self):

        return self

    def next(self):

        # Increment counters

        # Determine stopping criteria
        if self.num_steps is None:
            if (self.iter_batch + 1) > len(self.batch_order) or (self.iter_rewind + 1 > 50):
                raise StopIteration()
        else:
            if (self.iter_steps + 1) == self.num_steps:
                raise StopIteration()
            if (self.iter_batch + 1) == len(self.batch_order):
		load_not_ok = True
		self.num_batches = 0

		while load_not_ok:

	                if self.is_long==0:
        	            files = os.listdir(self.pathname)
			    file = random.choice(files)
			    
	                    print('New file. . .')    
		  
        	            if file[:4]=='long':
                	        self.is_long = 3                       

        		    self.filename = os.path.join(self.pathname,file)
			    self.validation_files.insert(0,self.filename)
			try:
	                    self.load()       
			except:
			    print('Error loading')

			if self.num_batches>0:
				load_not_ok = False

                    
        self.iter_batch += 1
        self.iter_steps += 1

        # Return relevant batch

        x, y, w = self.get_batch(self.iter_batch)
        return x, y, w

    def get_should_save(self):
        if self.should_save:
            self.should_save = False
            return True
        return False

    def rewind(self):

        # Reset iter
	if self.is_long==0:
        	self.iter_rewind += 1
        self.iter_batch = -1

        # Regular if not training
        if not self.is_training:
            self.batch_order = np.arange(self.num_batches)
            return
	else:
	    # Randomize order
            self.batch_order = np.random.permutation(self.num_batches)
            #self.batch_order = np.arange(self.num_batches)
	    self.should_save = False if self.iter_rewind%self.config.save_freq!=0 else True 
            print '(rewind data, shuffle)'

        # Remove those with perfect accuracy
        
        # Reset accuracy memory
        self.batch_accuracy = np.full(self.num_batches, 0, np.float32)
        self.batch_cost = np.full(self.num_batches, 0, np.float32)
        #self.confidence = np.full(sum(self.keep_these), 0, np.float32)

    def report_cost(self, accuracy, loss, confidence):

        # Save the accuracy in temporary memory
        current_batch = self.batch_order[self.iter_batch]
        self.batch_accuracy[current_batch] = accuracy
        self.batch_cost[current_batch] = loss
        
        ind = np.arange(current_batch*self.config.batch_size, (current_batch+1)*self.config.batch_size,
                        step=1,
                        dtype=np.int)
        #confidence = np.mean(np.reshape(confidence,[self.n_seg,self.config.batch_size]),axis=0)
        #self.confidence[ind] = confidence

    def get_batch(self, batch_num):

        # Find indices
        batch_num_ordered = self.batch_order[batch_num]
        ind = np.arange(batch_num_ordered*self.batch_size, (batch_num_ordered+1)*self.batch_size,
                        step=1,
                        dtype=np.int)

        # Find batches
        x = self.features[ind, :, :]
        t = self.logits[ind,:,:]
        t = np.reshape(t,[-1,self.config.num_classes])
        w = self.weights[ind,:]
        w = np.reshape(w,[-1])

        # Return
        return x, t, w

            
    def load(self):
        
        #self.update_load_list()

        # Import from CSV file
	print(self.is_long)
        if self.is_long>0:
            data_set = self.load_long()
        else:            
            data_set = self.load_h5()

        
        self.features = data_set.data
       
        assert np.round(self.num_batches, 0) == self.num_batches

        labels = np.transpose(data_set.target, axes=[1, 2, 0])    
        self.n_seg = labels.shape[0]//self.config.segsize
        print(labels.shape)
        self.num_batches = labels.shape[2] // self.batch_size

        print(self.num_batches)
	print(self.batch_size)
        labels = np.reshape(labels,[self.config.segsize, self.n_seg,self.config.num_classes,labels.shape[2]],order='F')
        self.logits = np.transpose(np.mean(labels,axis=0),axes=[2, 0, 1])


        # Weights
        weights = np.transpose(data_set.weights,axes = [1,0])
        weights = np.reshape(weights,[self.config.segsize, self.n_seg, weights.shape[1]],order='F')
        self.weights = np.transpose(np.mean(weights,axis=0),axes = [1,0])
                
        # Rewind
        self.rewind()
        

    def load_h5(self):

        # Read from file

        print(self.filename)

        f = h5py.File(self.filename,'r')
            
        dataT = f[u'/trainD']
        targetT = f[u'/trainL']
        weights = f[u'/trainW']
        print('%s loaded - Training' % (self.filename))
        
        print(dataT.shape)
        print(targetT.shape)
	print(weights.shape)

        self.batch_size = self.config.batch_size

        self.seq_in_file = dataT.shape[0]
        return Dataset(data=dataT, target=targetT, weights=weights)
    
    
    def load_long(self):   
        f = h5py.File(self.filename,'r')
        
        
            
        dataT = f[u'/trainD'+str(self.is_long)]
        targetT = f[u'/trainL'+str(self.is_long)]
        weights = np.ones(targetT.shape[0])
        print('%s loaded - Training' % (self.filename))
        
	excess = dataT.shape[0]%self.config.max_train_len

	dataT = dataT[:-excess,:]
	targetT = targetT[:-excess,:]
	weights = weights[:-excess]
	
	n_seg = dataT.shape[0]//self.config.max_train_len
	dataT = np.reshape(dataT,(self.config.max_train_len,n_seg,dataT.shape[1]),order='F')
        targetT = np.reshape(targetT,(self.config.max_train_len,n_seg,targetT.shape[1]),order='F')
	weights = np.reshape(weights,(self.config.max_train_len,n_seg))
	
	dataT = np.transpose(dataT,(1,0,2))
	targetT = np.transpose(targetT,(1,0,2))
	weights = np.transpose(weights,(1,0))

	print(dataT.shape)
	print(targetT.shape)

        self.batch_size = 1
        
        self.seq_in_file = dataT.shape[0]
        self.is_long -= 1
        return Dataset(data=dataT, target=targetT, weights=weights)

    def ismember(self, a, b):

        # Imitate ismember function from Matlab
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = True
        return np.array([bind.get(itm, False) for itm in a])
        
        
        
    def schedule_content(self):
	model_name = self.config.model_name
	print(model_name)
    	return '''#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --time=05:00:00
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=10000
#
#################

cd $HOME/SCmodel/python
ml python/2.7.5 hdf5/1.8.16 tensorflow/0.9.0

python sc_validate.py --model %s
''' % (model_name, model_name)


    def schedule(self):
        
        #Store filelist
        print(self.validation_files)              
        with open(self.config.model_name + 'validationlist.csv','wb') as csvfile:
        	wr = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_NONE)
        	for row in self.validation_files:
			print(row)
            		wr.writerow([row])
            
        self.validation_files = []        
        content = self.schedule_content()
        with tempfile.NamedTemporaryFile(delete=False) as job:
            job.write(content)
        command = 'sbatch %s'
        os.system(command % job.name)
    
