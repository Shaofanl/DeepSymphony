from DeepSymphony.common.HParam import HParam

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn, layers


class RefineNetHParam(HParam):
    # basic
    cells = [256]
    embed_dim = 200
    basic_cell = rnn.LSTMCell
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/RefineNet/'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # debug
    debug = False

    def __init__(self, **kwargs):
        self.register_check('vocab_size')
        self.register_check('output_size')

        super(RefineNetHParam, self).__init__(**kwargs)


class RefineNet(object):
    """
        Input: a sequence
        Output: a refined sequence
            or
        Input: a sequence
        Output: additional information (e.g. notes to duration)
    """
    def __init__(self, hparam):
        self.built = False
        if not hasattr(hparam, 'weight_path'):
            hparam.weight_path = \
                os.path.join(hparam.workdir, 'RefineNet.ckpt')

        self.hparam = hparam

    def build(self):
        hparam = self.hparam

        input = tf.placeholder("int32", [None, None], name="input")
        label = tf.placeholder("float32", [None, None], name="label")
        embeddings = tf.Variable(
            tf.random_uniform([hparam.vocab_size+1,
                               hparam.embed_dim],
                              -1.0, 1.0))
        input_emb = tf.nn.embedding_lookup(embeddings, input)

        encoder = rnn.MultiRNNCell([hparam.basic_cell(c)
                                    for c in hparam.cells])

        encoder_outputs, encoder_final_state = \
            tf.nn.dynamic_rnn(encoder, input_emb,
                              dtype=tf.float32)

        output = layers.linear(encoder_outputs,
                               1)  # hparam.output_size,)
        pred = output[:, :, 0]
        # output_sparse = tf.argmax(encoder_outputs, 2)
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=label, logits=output)
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean((output[:, :, 0]-label)**2)
        tf.summary.scalar('loss', loss)

        step_iter = tf.Variable(0, tf.int32)
        step_iter_op = step_iter.assign_add(1)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparam.learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparam.clip_norm)

        self.input = input
        self.label = label
        self.output = output
        self.pred = pred
        self.train_op = train_op
        self.step_iter = step_iter
        self.step_iter_op = step_iter_op
        self.loss = loss

        self.built = True

    def train(self, fetch_train_data, fetch_test_data, continued=None):
        if self.built is False:
            self.build()
        hparam = self.hparam

        if not continued:
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

            if continued:
                print 'restoring'
                saver.restore(sess, hparam.weight_path)
            else:
                sess.run(tf.global_variables_initializer())

            i = begin = self.step_iter.eval()
            while i < hparam.iterations+begin:
                input, label = fetch_train_data(hparam.batch_size)

                loss, summary, _ = sess.run(
                    [self.loss, merged, self.train_op],
                    feed_dict={self.input: input,
                               self.label: label})

                input, label = fetch_test_data(hparam.batch_size)
                valid_loss = sess.run(
                    self.loss,
                    feed_dict={self.input: input,
                               self.label: label})

                print(('Step {}: loss = {:.2f} | valid_loss = {:.2f}').
                      format(i, loss, valid_loss))
                train_writer.add_summary(summary, i)
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="valid_loss",
                                     simple_value=valid_loss)])
                train_writer.add_summary(summary, i)

                i += 1
                sess.run(self.step_iter_op)

            saver.save(sess, hparam.weight_path)

    def predict(self, seqs, max):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.hparam.weight_path)

            res = sess.run(self.pred,
                           feed_dict={self.input: seqs})
            res = np.clip(res, 1, max)
            res = np.round(res)
        return res


if __name__ == '__main__':
    hp = RefineNetHParam(vocab_size=10, output_size=10)
    m = RefineNet(hp)
    m.build()
