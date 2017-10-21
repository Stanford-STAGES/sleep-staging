from datetime import datetime
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import sc_network
import sc_config
import sc_reader

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('scope', 'oct', 'Which scope to train')
flags.DEFINE_string('model', 'oct_sh_ls_lstm', 'Which model to build')
flags.DEFINE_boolean('validate', False, 'Whether or not to schedule validation')
flags.DEFINE_integer('proceed', 1, 'Whether or not to continue previous')

def main(argv=None):

    if FLAGS.model[0:3] == 'oct':
        config = sc_config.OCConfig(restart=not(FLAGS.proceed>0), model_name=FLAGS.model, is_training=True)
    else:
        config = sc_config.ACConfig(restart=not(FLAGS.proceed>0), model_name=FLAGS.model, is_training=True)

    if gfile.Exists(config.train_dir):
        gfile.DeleteRecursively(config.train_dir)
        gfile.MakeDirs(config.train_dir)

    train(config)

def get_variable_names():
    """Returns list of all variable names in this model.

    Returns:
    List of names.
    """
    with tf.Graph.as_default():
        return [v.name for v in variables.trainable_variables()]


def train(config):

    train_data = sc_reader.ScoreData(config.train_data, config, num_steps=config.max_steps)

    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=False)) as session:


        m = sc_network.SCModel(config)
	print('Net configured')
        #s = tf.train.Saver(tf.all_variables(), max_to_keep=2)
        s = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        #print(tf.global_variables())
        #print(tf.all_variables())
        #summary_op = tf.merge_all_summaries()

        ckpt = tf.train.get_checkpoint_state(config.model_dir)

        if (FLAGS.proceed > 0 and not(not(ckpt))):
            print('Proceeding from checkpoint. . ')
            s.restore(session,ckpt.model_checkpoint_path)
            check_string = ckpt.model_checkpoint_path
            string_ind = check_string.find('ckpt') + 5
            train_data.iter_rewind += int(ckpt.model_checkpoint_path[string_ind:])

        else:
            #tf.initialize_all_variables().run()
	    
	    #init = tf.global_variables_initializer()
	    print('Initializing..')
	    #session.run(init)
	    #init.run()
	    session.run(tf.variables_initializer([v for v in tf.global_variables()]))
            print('Initialized!')

        #summary_writer = tf.train.SummaryWriter(config.train_dir, session.graph)

        for batch_input, batch_target, batch_weights in train_data:
            learning_rate = 0.005 * np.exp(-train_data.iter_rewind/750)
            step = train_data.iter_steps + 1
            print(learning_rate)
            print(train_data.batch_size)
            print(batch_input.shape)
            print(batch_target.shape)
            if not step % 30:
                _, loss, cross_ent, accuracy, baseline, confidence, softmax = session.run([m.train_op, m.loss, m.cross_ent, m.accuracy, m.baseline, m.confidence, m.softmax], feed_dict={
                    m.features: batch_input,
                    m.targets: batch_target,
                    m.mask: batch_weights,
                    m.batch_size: train_data.batch_size,
                    m.learning_rate: learning_rate
                    })
            else:
                _, loss, cross_ent, accuracy, baseline, confidence, softmax = session.run([m.train_op, m.loss, m.cross_ent, m.accuracy, m.baseline, m.confidence, m.softmax ], feed_dict={
                    m.features: batch_input,
                    m.targets: batch_target,
                    m.mask: batch_weights,
                    m.batch_size: train_data.batch_size,
                    m.learning_rate: learning_rate
                    })


	    #summary_writer.add_summary(summary_str, step)
            train_data.report_cost(accuracy, loss, confidence)
            if not 'cross_ent_r' in locals():
                cross_ent_r = cross_ent
                loss_r = loss
                accuracy_r = accuracy
                base_r = np.max(baseline)

            if len(train_data.validation_files)>50:
                train_data.schedule()
                print('Validation Scheduled')

            if not step % 1:
                now = datetime.now().time()
                print('%s | Iter %4d | Cross-Entropy= %.5f | Loss= %.5f | Train= %.5f | Baseline: %.5f' % (now, step, cross_ent, loss, accuracy, np.max(baseline)))
                print('Dist: %.3f, %.3f, %.3f, %.3f, %.3f' % (baseline[0], baseline[1], baseline[2], baseline[3], baseline[4]))
                cross_ent_r = cross_ent_r * 0.99 + 0.01 * cross_ent
                loss_r = loss_r * 0.99 + 0.01 * loss
                accuracy_r = accuracy_r * 0.99 + 0.01 * accuracy
                base_r = base_r * 0.99 + 0.01 * np.max(baseline)
                print('Running: | Cross-Entropy= %.5f | Loss= %.5f | Train= %.5f | Baseline: %.5f' % (cross_ent_r, loss_r, accuracy_r, base_r))

                #print((softmax[15:30,:]))
                #print((batch_target[15:30,:]))
            if train_data.get_should_save():
                s.save(session, config.checkpoint_file(), global_step=train_data.iter_rewind)
                if FLAGS.validate:
                    #sc_validate.schedule(config, train_data.iter_rewind)
                    print('(generated model checkpoint %.0f)' % train_data.iter_rewind)



if __name__ == '__main__':
    tf.app.run()





