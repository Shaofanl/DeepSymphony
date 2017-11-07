# !!!!!!!! code in this file is not optimized.
# check CondStackedLSTM for latest structure.
from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.RNN import rnn_wrapper

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.rnn as rnn
import shutil


class StackedLSTMHParam(HParam):
    # basic
    cells = [256, 256, 256]
    basic_cell = rnn.BasicLSTMCell
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/StackedLSTM'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # generate
    temperature = 1.0

    def __init__(self, **kwargs):
        self.register_check('timesteps')
        self.register_check('input_dim')
        self.register_check('output_dim')

        super(StackedLSTMHParam, self).__init__(**kwargs)


class StackedLSTM(object):
    def __init__(self, hparam):
        self.built = False
        self.hparam = hparam

    def build(self, mode):
        assert(self.built is False)
        assert(mode in ['train', 'generate'])
        hparam = self.hparam

        if mode == 'train':
            inputs = tf.placeholder("float", [hparam.batch_size,
                                              hparam.timesteps,
                                              hparam.input_dim],
                                    name="inputs")
            labels = tf.placeholder("int32", [hparam.batch_size,
                                              hparam.timesteps],
                                    name="labels")
        elif mode == 'generate':
            inputs = tf.placeholder("float", [hparam.batch_size,
                                              1,
                                              hparam.input_dim],
                                    name="inputs")

        with tf.variable_scope("LSTM"):
            # simple form
            cells, init_state, outputs, final_state = \
                    rnn_wrapper(inputs=inputs,
                                cells=hparam.cells,
                                basic_cell=hparam.basic_cell)

            # detailed form
            # cells = [hparam.basic_cell(c) for c in hparam.cells]
            # cells = rnn.MultiRNNCell(cells)
            # init_state = cells.zero_state(hparam.batch_size, tf.float32)

            # outputs, final_state = tf.nn.dynamic_rnn(
            #     cells, inputs,
            #     initial_state=init_state)

        with tf.variable_scope("top"):
            scores = tf.contrib.layers.linear(outputs,
                                              hparam.output_dim)

        if mode == 'train':
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=scores)
            loss = tf.reduce_mean(loss)

            optimizer = tf.train.AdamOptimizer(
                learning_rate=hparam.learning_rate)
            train_op = tf.contrib.slim.learning.create_train_op(
                loss, optimizer, clip_gradient_norm=hparam.clip_norm)
            tf.add_to_collection('train_op', train_op)
            tf.summary.scalar('loss', loss)

            self.train_op = train_op
            self.loss = loss
            self.labels = labels
        elif mode == 'generate':
            temperature = tf.placeholder(tf.float32, [], name="temperature")
            pred = tf.nn.softmax(tf.div(scores, temperature))

            self.pred = pred
            self.temperature = temperature
            self.init_state = init_state
            self.final_state = final_state

        self.inputs = inputs
        self.built = True

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

                _, loss, summary = sess.run([self.train_op, self.loss, merged],
                                            feed_dict={self.inputs: xi,
                                                       self.labels: yi})
                print('Step %d: loss = %.2f' % (i, loss))
                train_writer.add_summary(summary, i)
                i += 1

            saver = tf.train.Saver()
            saver.save(sess, os.path.join(hparam.workdir, 'StackedLSTM.ckpt'))

    def generate(self, handle, length=1000):
        if self.built is False:
            self.build(mode='generate')
        hparam = self.hparam
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, os.path.join(hparam.workdir,
                                             'StackedLSTM.ckpt'))

            results = []
            current = np.zeros((hparam.batch_size, 1, hparam.input_dim))
            state = sess.run(self.init_state)
            for i in range(length):
                feed_dict = {self.inputs: current,
                             self.temperature: hparam.temperature,
                             self.init_state: state}
                output, state = sess.run([self.pred, self.final_state],
                                         feed_dict=feed_dict)
                current = handle(output)
                results.append(current)
            return np.concatenate(results)
