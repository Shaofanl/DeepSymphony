from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.RNN import rnn_wrapper

import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.rnn as rnn
import shutil
from types import GeneratorType


class CondStackedLSTMHParam(HParam):
    # basic
    embed_dim = 500
    cells = [256, 256, 256]
    basic_cell = rnn.BasicLSTMCell
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    iterations = 300
    # logs
    workdir = './temp/CondStackedLSTM'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # generate
    temperature = 1.0

    def __init__(self, **kwargs):
        self.register_check('input_dim')
        self.register_check('output_dim')
        self.register_check('cond_len')

        super(CondStackedLSTMHParam, self).__init__(**kwargs)


class CondStackedLSTM(object):
    def __init__(self, hparam):
        if not hasattr(hparam, 'weight_path'):
            hparam.weight_path = \
                os.path.join(hparam.workdir, 'CondStackedLSTM.ckpt')

        self.built = False
        self.hparam = hparam

    def build(self):
        hparam = self.hparam

        inputs = tf.placeholder("int32", [None, None],
                                name="inputs")
        labels = tf.placeholder("int32", [None, None],
                                name="labels")
        cond = tf.placeholder("float", [None, None,
                                        hparam.cond_len])

        embeddings = tf.Variable(
            tf.random_uniform([hparam.input_dim,
                               hparam.embed_dim],
                              -1.0, 1.0))
        inputs_emb = tf.nn.embedding_lookup(embeddings, inputs)
        joint_input = tf.concat([inputs_emb, cond], 2)
        batch_size = tf.shape(inputs)[0]  # a tensor, not value

        with tf.variable_scope("LSTM"):
            # simple form
            cells, init_state, outputs, final_state = \
                    rnn_wrapper(inputs=joint_input,
                                batch_size=batch_size,
                                cells=hparam.cells,
                                basic_cell=hparam.basic_cell)

        with tf.variable_scope("top"):
            scores = tf.contrib.layers.linear(outputs,
                                              hparam.output_dim)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=scores)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparam.learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparam.clip_norm)
        tf.summary.scalar('loss', loss)

        iter_step = tf.Variable(0, name='iter_step')
        iter_step_op = iter_step.assign_add(1)

        # training
        self.inputs = inputs
        self.cond = cond
        self.train_op = train_op
        self.loss = loss
        self.labels = labels
        self.iter_step_op = iter_step_op
        self.iter_step = iter_step
        self.batch_size = batch_size

        # generate
        temperature = tf.placeholder(tf.float32, [], name="temperature")
        prob = tf.nn.softmax(tf.div(scores, temperature))

        self.pred = prob
        self.temperature = temperature
        self.init_state = init_state
        self.final_state = final_state
        self.built = True

    def train(self, fetch_data):
        if not self.built:
            self.build()
        hparam = self.hparam

        if not hparam.continued:
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
            saver = tf.train.Saver()

            if hparam.continued:
                print 'restoring'
                saver.restore(sess, hparam.weight_path)
            else:
                sess.run(tf.global_variables_initializer())

            i = begin = self.iter_step.eval()
            while i < begin+hparam.iterations:
                xi, yi, ci = fetch_data(hparam.batch_size)

                _, loss, summary = sess.run(
                    [self.train_op, self.loss, merged],
                    feed_dict={self.inputs: xi,
                               self.labels: yi,
                               self.cond: ci})

                print('Step %d: loss = %.2f' % (i, loss))
                train_writer.add_summary(summary, i)

                i += 1
                sess.run(self.iter_step_op)
            saver.save(sess, hparam.weight_path)

    def generate(self, conds, sample, length=1000):
        if not self.built:
            self.build()
        hparam = self.hparam

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, hparam.weight_path)

            results = []
            event = np.random.randint(hparam.output_dim)
            state = sess.run(self.init_state,
                             feed_dict={self.batch_size: 1})
            for i in range(length):
                if isinstance(conds, GeneratorType):
                    condi = conds.next()
                else:
                    condi = conds[i]
                feed_dict = {self.inputs: [[event]],
                             self.cond: [[condi]],
                             self.temperature: hparam.temperature,
                             self.init_state: state}
                prob, state = sess.run(
                    [self.pred, self.final_state],
                    feed_dict=feed_dict)
                event = sample(prob[0][0])
                results.append(event)
            return np.array(results)
