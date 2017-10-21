
                
                
import numpy as np
import sc_network
import sc_config
import sc_reader_validate

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
        config = sc_config.ACConfig(model_name=FLAGS.model, is_training=True)

    test(config)
    
def test(config):
    
    validation_data = sc_reader_validate.ScoreData(config.train_data, config)
    while validation_data.anyleft:

	with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:

		m = sc_network.SCModel(config)
		s = tf.train.Saver(tf.all_variables())
	        summary_op = tf.merge_all_summaries()
				
       	        ckpt = tf.train.get_checkpoint_state(config.model_dir)
       	        s.restore(session, ckpt.model_checkpoint_path)
		print(ckpt.model_checkpoint_path)
               	check_string = ckpt.model_checkpoint_path
		string_ind = check_string.find('ckpt') + 5
		validation_data.iter_rewind = int(ckpt.model_checkpoint_path[string_ind:])

               	for batch_input, batch_target in validation_data:
				print(str(validation_data.iter_batch)+' of '+str(validation_data.num_batches))				
				batch_target = np.squeeze(batch_target)
				if (validation_data.iter_batch==0 or not(validation_data)) and config.lstm:
					state = np.zeros([np.ones([1]),config.num_hidden*2])
							   
				if np.rank(batch_input)==2:
					batch_input = np.expand_dims(batch_input,0)
				        	
				if config.lstm:
                        	
					"""					
					loss, cross_ent, accuracy, baseline, state = session.run([m.loss, m.cross_ent, m.accuracy, m.baseline, m.final_state], feed_dict={
						m.features: batch_input,
						m.targets: batch_target,
						m.mask: np.ones(len(batch_target)),
						m.batch_size: 10
						})

					"""
			                loss, cross_ent, accuracy, baseline, confidence, summary_str, softmax = session.run([m.loss, m.cross_ent, m.accuracy, m.baseline, m.confidence, summary_op, m.softmax ], feed_dict={
                				m.features: batch_input,
					        m.targets: batch_target,
				                m.mask: np.ones(len(batch_target)),
						m.batch_size: 1,
						m.learning_rate: 0
	                    })

					
				else:
					loss, cross_ent, accuracy, baseline = session.run([m.loss, m.cross_ent, m.accuracy, m.baseline], feed_dict={
						m.features: batch_input,
						m.targets: batch_target,
						m.mask: np.ones(len(batch_target)),
						m.batch_size: np.ones([1])
						})
				print('Acc')	
				print(accuracy)		
				print(baseline)
				print('Loss')
				print(loss)
				validation_data.record_results(loss,cross_ent,accuracy,baseline)

    
if __name__ == '__main__':
    #tf.app.run()
