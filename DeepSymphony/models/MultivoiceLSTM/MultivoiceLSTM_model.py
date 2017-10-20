from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.RNN import rnn_wrapper

import os
import shutil
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


class MultivoiceLSTMHParam(HParam):
    # basic
    nb_voices = 2
    b_cells = [256]  # bottom cells
    m_cells = [256]  # middle cells
    t_cells = [256]  # top cells
    basic_cell = rnn.BasicLSTMCell
    input_dim = output_dim = 128
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/MutlivoiceLSTM'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False

    def __init__(self, **kwargs):
        self.register_check('timesteps')

        super(MultivoiceLSTMHParam, self).__init__(**kwargs)


class MultivoiceLSTM(object):
    def __init__(self, hparam):
        self.built = False
        self.hparam = hparam

    def build(self, mode):
        assert(self.built is False)
        assert(mode in ['train'])
        hparam = self.hparam

        inputs = [tf.placeholder("float", [hparam.batch_size,
                                           hparam.timesteps,
                                           hparam.input_dim],
                                 name="inputs/v_{}".format(i))
                  for i in range(hparam.nb_voices)]
        labels = [tf.placeholder("float", [hparam.batch_size,
                                           hparam.timesteps,
                                           hparam.input_dim],
                                 name="labels/v_{}".format(i))
                  for i in range(hparam.nb_voices)]

        # for generation
        init_states = []
        final_states = []

        bot_outputs = []
        with tf.variable_scope("bottom"):
            for vind in range(hparam.nb_voices):
                with tf.variable_scope("voice_{}".format(vind)):
                    cells, init_state, outputs, final_state = \
                        rnn_wrapper(inputs=inputs[vind],
                                    cells=hparam.b_cells,
                                    basic_cell=hparam.basic_cell)
                    bot_outputs.append(outputs)
                    init_states.append(init_state)
                    final_states.append(final_state)

        with tf.variable_scope("middle"):
            joint = tf.concat(bot_outputs, axis=2)

            cells, init_state, mid_outputs, final_state = \
                rnn_wrapper(inputs=joint,
                            cells=hparam.b_cells,
                            basic_cell=hparam.basic_cell)
            init_states.append(init_state)
            final_states.append(final_state)

        top_outputs = []
        with tf.variable_scope("top"):
            for vind in range(hparam.nb_voices):
                with tf.variable_scope("voice_{}".format(vind)):
                    cells, init_state, outputs, final_state = \
                        rnn_wrapper(inputs=inputs[vind],
                                    cells=hparam.b_cells,
                                    basic_cell=hparam.basic_cell)
                    outputs = tf.contrib.layers.linear(outputs,
                                                       hparam.output_dim)
                    top_outputs.append(outputs)

                    init_states.append(init_state)
                    final_states.append(final_state)

        if mode == 'train':
            losses = []
            for labels_i, top_outputs_i in zip(labels, top_outputs):
                loss_i = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=labels_i, logits=top_outputs_i)
                loss_i = tf.reduce_mean(tf.reduce_sum(loss_i, axis=2))
                losses.append(loss_i)
            total_loss = tf.reduce_mean(losses)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=hparam.learning_rate)
            train_op = tf.contrib.slim.learning.create_train_op(
                total_loss, optimizer, clip_gradient_norm=hparam.clip_norm)
            tf.summary.scalar('loss', total_loss)
            for vind in range(hparam.nb_voices):
                tf.summary.scalar('loss/voice_{}'.format(vind),
                                  losses[vind])

            self.train_op = train_op
            self.inputs = inputs
            self.labels = labels
            self.loss = total_loss

    def train(self, fetch_data):
        if self.built is False:
            self.build(mode='train')
        hparam = self.hparam

        if os.path.exists(hparam.workdir):
            if hparam.overwrite_workdir:
                shutil.rmtree(hparam.workdir)
            else:
                raise Exception("The workdir exists.")
        os.makedirs(hparam.workdir)
        os.makedirs(os.path.join(hparam.workdir, hparam.tensorboard_dir))

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(
                os.path.join(hparam.workdir, hparam.tensorboard_dir),
                sess.graph)
            merged = tf.summary.merge_all()
            sess.run(tf.global_variables_initializer())

            i = 0
            while i < hparam.iterations:
                xi, yi = fetch_data(hparam.batch_size)
                feed_dict = dict(zip(self.inputs, xi)+zip(self.labels, yi))

                _, loss, summary = sess.run([self.train_op, self.loss, merged],
                                            feed_dict=feed_dict)
                print('Step %d: loss = %.2f' % (i, loss))
                train_writer.add_summary(summary, i)
                i += 1

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(hparam.workdir,
                                          'MultivoiceLSTM.ckpt'))
