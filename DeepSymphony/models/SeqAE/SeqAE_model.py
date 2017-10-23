from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.Helpers import GreedyEmbeddingDecisionHelper

import os
import shutil

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq, layers


class SeqAEHParam(HParam):
    # basic
    encoder_cells = [256]
    decoder_cells = [256]
    embed_dim = 200
    basic_cell = rnn.LSTMCell
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/SeqAE/'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # debug
    debug = False

    def __init__(self, **kwargs):
        self.register_check('timesteps')
        self.register_check('vocab_size')

        super(SeqAEHParam, self).__init__(**kwargs)
        assert(self.encoder_cells == self.decoder_cells)
        # in next version, maybe we can have
        #   encoder: layer1 layer2 layer3
        #                            v
        #                    final_state(code)
        #                            v
        #                 decoder: layer1 layer2 layer3


class SeqAE(object):
    """
        Sequential Autoencoder
          compress the sequence into a code

        seq = (A, B, C)
        Encoder:
            rnn   rnn   rnn --> code
             A     B     C

        Decoder:
            Training: (TrainingHelper)
                        A     B     C
                code-> rnn   rnn   rnn
                     <start>  A     B

            Testing: (GreedyEmbeddingHelper)
                       y_1   y_2   y_3
              random-> rnn   rnn   rnn
                     <start> y_1   y_2

    """
    def __init__(self, hparam):
        self.built = False
        if not hasattr(hparam, 'weight_path'):
            hparam.weight_path = \
                os.path.join(hparam.workdir, 'SeqAE.ckpt')

        # hparam.encoder_cells.append(hparam.vocab_size+2)
        # hparam.decoder_cells.append(hparam.vocab_size+2)

        self.hparam = hparam
        self.start_token = hparam.vocab_size
        # since we are using fixed sequence here,
        # we do not add eos_token to the data
        self.eos_token = hparam.vocab_size+1

    def build(self):
        hparam = self.hparam

        seqs = tf.placeholder("int32", [hparam.batch_size,
                                        hparam.timesteps],
                              name="seqs")
        start_tokens = tf.ones([hparam.batch_size],
                               dtype=tf.int32) * self.start_token
        embeddings = tf.Variable(
            tf.random_uniform([hparam.vocab_size+2,
                               hparam.embed_dim],
                              -1.0, 1.0))

        # pad_seqs: (bs, 1+ts)
        pad_seqs = tf.concat([tf.expand_dims(start_tokens, 1), seqs], 1)

        # pad_seqs_emb: (bs, 1+ts, dim)
        pad_seqs_emb = tf.nn.embedding_lookup(embeddings, pad_seqs)

        encoder = rnn.MultiRNNCell([hparam.basic_cell(c)
                                    for c in hparam.encoder_cells])
        encoder_outputs, encoder_final_state = \
            tf.nn.dynamic_rnn(encoder,
                              pad_seqs_emb[:, 1:, :],  # (A, B, C)
                              dtype=tf.float32)
        code = encoder_final_state

        train_helper = tf.contrib.seq2seq.TrainingHelper(
            pad_seqs_emb[:, :-1, :],  # (<start>, A, B)
            tf.ones([hparam.batch_size], tf.int32) * hparam.timesteps)
        # this code use the output of RNN directly, while we add an additional
        #   decision layer on the top of RNN.
        # pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        #     embeddings,
        #     start_tokens=start_tokens,
        #     end_token=self.eos_token)
        pred_helper = GreedyEmbeddingDecisionHelper(
            decision_scope='decision',
            reuse=True,
            output_dim=hparam.vocab_size+2,
            embedding=embeddings,
            start_tokens=start_tokens,
            end_token=self.eos_token)

        def decode(helper, scope, initial_state, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                cells = rnn.MultiRNNCell([hparam.basic_cell(c)
                                          for c in hparam.decoder_cells])

                decoder = seq2seq.BasicDecoder(
                    cell=cells, helper=helper,
                    initial_state=initial_state)
                final_outputs, final_state, final_sequence_lengths = \
                    seq2seq.dynamic_decode(
                        decoder=decoder, output_time_major=False,
                        maximum_iterations=hparam.timesteps,
                        # impute_finished=True,
                    )
                scores = layers.linear(final_outputs.rnn_output,
                                       hparam.vocab_size+2,
                                       scope='decoder/decision')
                pred = tf.argmax(scores, axis=2)
                return scores, pred
        scores, train_pred, = decode(train_helper,
                                     'decode',
                                     initial_state=code)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=seqs,
            logits=scores)
        loss = tf.reduce_mean(loss)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparam.learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparam.clip_norm)
        tf.summary.scalar('loss', loss)

        # same decoder, but use different helper
        pred_scores, pred = decode(pred_helper,
                                   'decode',
                                   initial_state=code,
                                   reuse=True)

        # train
        self.seqs = seqs
        self.loss = loss
        self.train_op = train_op
        # debug
        self.pad_seqs_emb = pad_seqs_emb
        self.embeddings = embeddings
        self.train_pred = train_pred
        # generate
        self.code = code
        self.pred = pred
        self.built = True

    def train(self, fetch_data, continued=None):
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

            # debug
            if hparam.debug:
                seqs = fetch_data(hparam.batch_size)
                code, pad_seqs_emb, embeddings, loss =\
                    sess.run([self.code,
                              self.pad_seqs_emb,
                              self.embeddings,
                              self.loss],
                             feed_dict={self.seqs: seqs})
                print code
                print pad_seqs_emb
                print embeddings
                import ipdb
                ipdb.set_trace()

            global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
            i = begin = global_step.eval()
            while i < hparam.iterations+begin:
                seqs = fetch_data(hparam.batch_size)

                _, loss, summary = sess.run([self.train_op,
                                             self.loss,
                                             merged],
                                            feed_dict={self.seqs: seqs})

                print('Step %d: loss = %.2f' % (i, loss))
                train_writer.add_summary(summary, i)
                i += 1

            saver.save(sess, hparam.weight_path)

    def collect(self, fetch_data, samples=10):
        hparam = self.hparam
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, hparam.weight_path)

            collection = []
            seqs = []
            for _ in range(samples):
                seq = fetch_data(hparam.batch_size)
                code = sess.run([self.code],
                                feed_dict={self.seqs: seq})
                collection.append(code)
                seqs.append(seq)
            return collection, seqs

    def generate(self, code):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.hparam.weight_path)

            prediction = sess.run(self.pred,
                                  feed_dict={self.code: code})
            return prediction

    def eval(self, seqs):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.hparam.weight_path)

            test_pred, train_pred = \
                sess.run([self.pred, self.train_pred],
                         feed_dict={self.seqs: seqs})
            return test_pred, train_pred
