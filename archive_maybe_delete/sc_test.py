import numpy as np
import sc_network
import sc_config
import sc_reader_test

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('scope', 'oct', 'Which scope to test')
flags.DEFINE_string('model', 'oct_sh_ls_lstm', 'Which model to test')
flags.DEFINE_integer('checknr', 0, 'Whether or not to continue previous')

def main(argv=None):

	if FLAGS.model[0:3] == 'oct':
		config = sc_config.OCConfig(model_name=FLAGS.model, is_training=False)
	else:
        	config = sc_config.ACConfig(model_name=FLAGS.model, is_training=False)

	test(config)
    
def test(config):
    
	test_data = sc_reader_test.ScoreData(config)
	while test_data.anyleft:

		with tf.Graph().as_default() as g:

			m = sc_network.SCModel(config)

			s = tf.train.Saver(tf.all_variables())
			
			with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:
				print(config.model_dir_test)
				ckpt = tf.train.get_checkpoint_state(config.model_dir_test)
				print(ckpt)
				s.restore(session, ckpt.model_checkpoint_path)
			

				try:
					for batch_input, batch_target in test_data:
						print(str(test_data.iter_batch)+' of '+str(test_data.num_batches))
						if batch_target.shape[0]==0:
							test_data.record_results([])
							continue						
					
						if test_data.iter_batch==0 and config.lstm:
							state = np.zeros([np.ones([1]),config.num_hidden*2])
							   
					
						if config.lstm:
							prediction, state = session.run([m.logits, m.final_state], feed_dict={
							m.features: batch_input,
							m.targets: batch_target,
							m.mask: np.ones(len(batch_target)),
							m.batch_size: np.ones([1]),
							m.initial_state: state                        
							})
		
						else:
							prediction = session.run([m.logits], feed_dict={
							m.features: batch_input,
							m.targets: batch_target,
							m.mask: np.ones(len(batch_target)),
							m.batch_size: np.ones([1])
							})
							
						test_data.record_results(prediction)
	
				except (RuntimeError, TypeError, NameError, IOError):
					print('error!')
if __name__ == '__main__':
	tf.app.run()
