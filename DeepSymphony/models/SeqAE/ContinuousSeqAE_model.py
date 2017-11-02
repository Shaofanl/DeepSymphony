# experimental
from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.Helpers import GreedyEmbeddingDecisionHelper

import os
import shutil

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq, layers


class ContinuousSeqAEHParam(HParam):
    # basic
    encoder_cells = [256]
    decoder_cells = [256]
    embed_dim = 200
    basic_cell = rnn.LSTMCell
    alpha = 1.0
    # training
    learning_rate = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/ContinuousSeqAE/'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # debug
    debug = False

    def __init__(self, **kwargs):
        self.register_check('timesteps')
        self.register_check('gen_timesteps')
        self.register_check('vocab_size')

        super(ContinuousSeqAEHParam, self).__init__(**kwargs)
        assert(self.encoder_cells[-1] == self.decoder_cells[0])
        # in next version, maybe we can have
        #   encoder: layer1 layer2 layer3
        #                            v
        #                    final_state(code)
        #                            v
        #                 decoder: layer1 layer2 layer3
        if self.basic_cell == rnn.LSTMCell:
            raise NotImplemented("LSTM has two states,\
                                 which might confuse the training")


class ContinuousSeqAE(object):
    """
        Sequential Autoencoder
          compress the sequence into a code

        seq = (A, B, C)
        Encoder:
            rnn   rnn   rnn --> code
             A     B     C

        Decoder:
            Training: (TrainingHelper)
                       y_1   y_2   y_3
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
                os.path.join(hparam.workdir, 'ContinuousSeqAE.ckpt')

        # hparam.encoder_cells.append(hparam.vocab_size+2)
        # hparam.decoder_cells.append(hparam.vocab_size+2)

        self.hparam = hparam
        self.start_token = hparam.vocab_size
        # since we are using fixed sequence here,
        # we do not add eos_token to the data
        self.eos_token = hparam.vocab_size+1

    def build(self):
        hparam = self.hparam

        seqs = tf.placeholder("int32", [None,
                                        hparam.timesteps],
                              name="seqs")
        tri_source_seqs = tf.placeholder("int32", [None,
                                                   hparam.timesteps],
                                         name="tri_source_seqs")
        tri_neighbour_seqs = tf.placeholder("int32", [None,
                                                      hparam.timesteps],
                                            name="tri_neighbour_seqs")
        tri_intruder_seqs = tf.placeholder("int32", [None,
                                                     hparam.timesteps],
                                           name="tri_intruder_seqs")

        start_tokens = tf.ones([tf.shape(seqs)[0]],
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

        def code_active(x):
            # cell is the output of
            # TODO: assertion
            return x
            # return tf.minimum(tf.maximum(1., hparam.beta*x), -1.)
            # return tf.nn.tanh(x*hparam.beta)
            # return tf.nn.sigmoid(x*hparam.beta)

        def quantize_code(x):
            return tf.sign(x)
            # return tf.round(x*10)/10.
            # return tf.round(x*2)/2.
        quantized_max = 1.0
        quantized_min = -1.0
        quantized_range = (quantized_max - quantized_min)**2

        ori_code = encoder_final_state[-1]
        code = code_active(ori_code)
        quantized_code = quantize_code(code)  # *tf.reduce_mean(tf.abs(code))
        decoder_start_tokens = tf.ones([tf.shape(quantized_code)[0]],
                                       dtype=tf.int32) * self.start_token
        q_loss = tf.reduce_mean(tf.reduce_sum(
            tf.square(quantized_code-ori_code), 1))

        # triple
        tri_src_emb = tf.nn.embedding_lookup(embeddings, tri_source_seqs)
        tri_nei_emb = tf.nn.embedding_lookup(embeddings, tri_neighbour_seqs)
        tri_int_emb = tf.nn.embedding_lookup(embeddings, tri_intruder_seqs)

        _, tri_src_code = tf.nn.dynamic_rnn(encoder,
                                            tri_src_emb,
                                            dtype=tf.float32)
        _, tri_nei_code = tf.nn.dynamic_rnn(encoder,
                                            tri_nei_emb,
                                            dtype=tf.float32)
        _, tri_int_code = tf.nn.dynamic_rnn(encoder,
                                            tri_int_emb,
                                            dtype=tf.float32)
        tri_src_code = code_active(tri_src_code[-1])
        tri_nei_code = code_active(tri_nei_code[-1])
        tri_int_code = code_active(tri_int_code[-1])

        quantized_tri_src_code = quantize_code(tri_src_code)
        quantized_tri_nei_code = quantize_code(tri_nei_code)
        quantized_tri_int_code = quantize_code(tri_int_code)

        def con_loss_f(src, nei, int):
            nei_dist = tf.reduce_sum(tf.square(src - nei), 1)
            # int_dist = tf.reduce_sum(tf.square(src - int), 1)
            # con_loss = tf.reduce_mean(tf.maximum(0., nei_dist-int_dist+1))
            con_loss = tf.reduce_mean(tf.maximum(0., nei_dist-quantized_range))
            return con_loss
        con_loss = con_loss_f(tri_src_code,
                              quantized_tri_nei_code,
                              None) +\
                   con_loss_f(quantized_tri_src_code,
                              tri_nei_code,
                              None)


        # con_loss = con_loss_f(tri_src_code,
        #                       tri_nei_code,
        #                       tri_int_code)
        quantized_con_loss = tf.reduce_mean(tf.reduce_sum(
            tf.abs(quantized_tri_src_code-quantized_tri_nei_code) /
            quantized_range,
            1
        ))

        train_helper = tf.contrib.seq2seq.TrainingHelper(
            pad_seqs_emb[:, :-1, :],  # (<start>, A, B)
            tf.ones([tf.shape(seqs)[0]], tf.int32) * hparam.timesteps)
        quantized_train_helper = tf.contrib.seq2seq.TrainingHelper(
            pad_seqs_emb[:, :-1, :],  # (<start>, A, B)
            tf.ones([tf.shape(seqs)[0]], tf.int32) * hparam.timesteps)
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
            start_tokens=decoder_start_tokens,
            end_token=self.eos_token)

        def decode(helper, scope, code, timesteps, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                cells = rnn.MultiRNNCell([hparam.basic_cell(c)
                                          for c in hparam.decoder_cells])
                # create state
                init_state = cells.zero_state(tf.shape(code)[0], tf.float32)
                init_state = (code,)+init_state[1:]  # replace the 1st layer

                decoder = seq2seq.BasicDecoder(
                    cell=cells, helper=helper,
                    initial_state=init_state)
                final_outputs, final_state, final_sequence_lengths = \
                    seq2seq.dynamic_decode(
                        decoder=decoder, output_time_major=False,
                        maximum_iterations=timesteps,
                        # impute_finished=True,
                    )
                scores = layers.linear(final_outputs.rnn_output,
                                       hparam.vocab_size+2,
                                       scope='decoder/decision')
                pred = tf.argmax(scores, axis=2)
                return scores, pred
        scores, train_pred, = decode(train_helper,
                                     'decode',
                                     code=code,
                                     timesteps=hparam.timesteps)
        quantized_scores, quantized_train_pred, = \
            decode(quantized_train_helper,
                   'decode',
                   code=quantized_code,
                   timesteps=hparam.timesteps,
                   reuse=True)

        rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=seqs,
            logits=scores)
        rec_loss = tf.reduce_mean(rec_loss)
        quantized_rec_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=seqs,
            logits=quantized_scores)
        quantized_rec_loss = tf.reduce_mean(quantized_rec_loss)
        loss = quantized_rec_loss + rec_loss +\
            hparam.alpha*(con_loss) + hparam.gamma*q_loss
        # +quantized_con_loss)

        optimizer = tf.train.AdamOptimizer(
            learning_rate=hparam.learning_rate)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparam.clip_norm)
        quantized_rec_train_op = tf.contrib.slim.learning.create_train_op(
            quantized_rec_loss, optimizer, clip_gradient_norm=hparam.clip_norm)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('rec_loss', rec_loss)
        tf.summary.scalar('q_rec_loss', quantized_rec_loss)
        tf.summary.scalar('q_con_loss', quantized_con_loss)
        tf.summary.scalar('con_loss', con_loss)

        # same decoder, but use different helper
        pred_scores, pred = decode(pred_helper,
                                   'decode',
                                   code=quantized_code,
                                   timesteps=hparam.gen_timesteps,
                                   reuse=True)

        # train
        self.seqs = seqs
        self.loss = loss
        self.con_loss = con_loss
        self.rec_loss = rec_loss
        self.quantized_rec_loss = quantized_rec_loss
        self.quantized_con_loss = quantized_con_loss
        self.train_op = train_op
        self.quantized_rec_train_op = quantized_rec_train_op 
        # debug
        self.pad_seqs_emb = pad_seqs_emb
        self.embeddings = embeddings
        self.train_pred = train_pred
        self.tri_src_code, self.tri_nei_code, self.tri_int_code = \
            tri_src_code, tri_nei_code, tri_int_code
        # tri
        self.tri_src_seqs = tri_source_seqs
        self.tri_nei_seqs = tri_neighbour_seqs
        self.tri_int_seqs = tri_intruder_seqs
        # generate
        self.ori_code = ori_code
        self.code = code
        self.quantized_code = quantized_code
        self.pred = pred
        self.built = True

    def train(self, fetch_data, fetch_tri_data, continued=None):
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
                src_seqs, nei_seqs, int_seqs = \
                    fetch_tri_data(hparam.batch_size)

                feed_dict = {self.tri_src_seqs: src_seqs,
                             self.tri_nei_seqs: nei_seqs,
                             self.tri_int_seqs: int_seqs}

                nei_dist, int_dist, src_code, nei_code, int_code =\
                    sess.run([self.nei_dist,
                              self.int_dist,
                              self.tri_src_code,
                              self.tri_nei_code,
                              self.tri_int_code],
                             feed_dict=feed_dict)
                import ipdb
                ipdb.set_trace()

            global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
            i = begin = global_step.eval()
            while i < hparam.iterations+begin:
                seqs = fetch_data(hparam.batch_size)
                tri_src_seqs, tri_nei_seqs, tri_int_seqs = \
                    fetch_tri_data(hparam.batch_size)

                feed_dict = {self.seqs: seqs,
                             self.tri_src_seqs: tri_src_seqs,
                             self.tri_nei_seqs: tri_nei_seqs,
                             self.tri_int_seqs: tri_int_seqs}

                if hparam.only_train_quantized_rec:
                    train_op = self.quantized_rec_train_op
                else:
                    train_op = self.train_op
                _, loss, con_loss, rec_loss, \
                    q_rec_loss, q_con_loss, summary = \
                    sess.run([train_op,
                              self.loss,
                              self.con_loss,
                              self.rec_loss,
                              self.quantized_rec_loss,
                              self.quantized_con_loss,
                              merged],
                             feed_dict=feed_dict)

                print(('Step {}: ' +
                       'loss = {:.2f} |' +
                       ' rec_loss = {:.2f} |' +
                       ' q_res_loss = {:.2f} |' +
                       ' con_loss = {:.2f} |' +
                       ' q_con_loss = {:.2f}').
                      format(i, loss, rec_loss, q_rec_loss,
                             con_loss, q_con_loss))
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
                import ipdb
                ipdb.set_trace()
                collection.append(code)
                seqs.append(seq)
            return collection, seqs

    def generate(self, code, quantized=False):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.hparam.weight_path)

            code_tensor = self.quantized_code if quantized else self.code
            prediction = sess.run(self.pred,
                                  feed_dict={code_tensor: code})
            return prediction

    def eval(self, seqs):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.hparam.weight_path)

            test_pred, train_pred = \
                sess.run([self.pred, self.train_pred],
                         feed_dict={self.seqs: seqs})
            return test_pred, train_pred

    def encode(self, batch, quantized=False):
        hparam = self.hparam
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, hparam.weight_path)

            code_tensor = self.quantized_code if quantized else self.code
            code = sess.run(code_tensor,
                            feed_dict={self.seqs: batch})
        return code
