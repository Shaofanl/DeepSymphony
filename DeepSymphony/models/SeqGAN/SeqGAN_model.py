from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.Helpers import GreedyEmbeddingHelper
from DeepSymphony.common.resnet import ResNetBuilder

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn, layers, slim


class SeqGANHParam(HParam):
    # basic
    cells = [256]
    basic_cell = rnn.GRUCell
    embed_dim = 500
    code_dim = 200
    # training
    D_lr = 1e-5
    G_lr = 1e-5
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/SeqGAN/'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # debug
    debug = False

    def __init__(self, **kwargs):
        self.register_check('timesteps')
        self.register_check('vocab_size')

        super(SeqGANHParam, self).__init__(**kwargs)


class SeqGAN(object):
    def __init__(self, hparam):
        self.built = False
        if not hasattr(hparam, 'weight_path'):
            hparam.weight_path = \
                os.path.join(hparam.workdir, 'SeqGAN.ckpt')
        self.hparam = hparam
        # self.start_token = hparam.vocab_size
        # self.eos_token = hparam.vocab_size+1

    def build(self):
        hparam = self.hparam

        code = tf.placeholder("float32", [None,
                                          hparam.code_dim],
                              name='code')
        real_seq = tf.placeholder("int32", [None,
                                            hparam.timesteps],
                                  name='real_seq')
        real_seq_img = tf.one_hot(real_seq, hparam.vocab_size)
        dis_train = tf.placeholder('bool', name='is_train')
        bs = tf.shape(code)[0]

        # generator
        with tf.variable_scope('generator'):
            cells = rnn.MultiRNNCell([hparam.basic_cell(c)
                                      for c in hparam.cells])
            # play with code
            # code_with_timestep = tf.concat(
            #     [
            #         tf.tile(tf.expand_dims(code, 1),
            #                 (1, hparam.timesteps, 1)),
            #         tf.tile(tf.expand_dims(tf.eye(hparam.timesteps), 0),
            #                 (bs, 1, 1)),
            #     ], -1
            # )
            code_with_timestep = tf.concat(
                [
                    tf.tile(tf.expand_dims(code, 1),
                            (1, hparam.timesteps, 1)),
                    tf.tile(tf.expand_dims(tf.expand_dims(tf.lin_space(
                        0., 1., hparam.timesteps), 0), -1),
                            (bs, 1, 1)),
                ], -1
            )
            outputs, states = tf.nn.dynamic_rnn(
                cells,
                code_with_timestep,
                dtype=tf.float32,
            )
            # fake_seq_img = tf.nn.softmax(layers.linear(
            #    outputs[:, :, :], hparam.vocab_size), -1)
            fake_seq_img = outputs
            fake_seq_img = tf.nn.softmax(fake_seq_img)
            fake_seq = tf.argmax(fake_seq_img, -1)

        # discriminator
        def dis(seq_img, bn_scope, reuse=False):
            with tf.variable_scope('discriminator', reuse=reuse):
                # x = tf.nn.embedding_lookup(embeddings, seq)
                x = tf.expand_dims(seq_img, -1)
                x = ResNetBuilder(dis_train,
                                  bn_scopes=['fake', 'real'],
                                  bn_scope=bn_scope).\
                    resnet(x, structure=[2, 2, 2, 2], filters=4, nb_class=1)
                x = tf.nn.sigmoid(x)
            return x
        # opt
        # problematic with the reuse bn
        fake_dis_pred = dis(fake_seq_img, bn_scope='fake')
        real_dis_pred = dis(real_seq_img, bn_scope='real', reuse=True)

        G_loss = tf.reduce_mean(tf.log(fake_dis_pred))
        D_loss = tf.reduce_mean(tf.log(real_dis_pred)) +\
            tf.reduce_mean(tf.log(1-fake_dis_pred))

        G_opt = tf.train.AdamOptimizer(learning_rate=hparam.G_lr,
                                       beta1=0.5, beta2=0.9)
        D_opt = tf.train.GradientDescentOptimizer(learning_rate=hparam.D_lr)
        D_iter = tf.Variable(0, name='D_iter')
        G_iter = tf.Variable(0, name='G_iter')
        G_train_op = slim.learning.create_train_op(
            G_loss, G_opt,
            variables_to_train=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, "generator"),
            global_step=G_iter,
            clip_gradient_norm=hparam.G_clipnorm
        )
        D_train_op = slim.learning.create_train_op(
            D_loss, D_opt,
            variables_to_train=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                "discriminator"),
            global_step=D_iter,
        )
        iter_step = tf.Variable(0, name='iter_step')
        iter_step_op = iter_step.assign_add(1)

        # input
        self.code = code
        self.real_seq = real_seq
        # summary
        self.summary_fake_img = tf.summary.image(
            'fake_img', tf.expand_dims(fake_seq_img, -1))
        self.summary_real_img = tf.summary.image(
            'real_img', tf.expand_dims(real_seq_img, -1))
        self.summary_G_loss = tf.summary.scalar('G_loss', G_loss)
        self.summary_D_loss = tf.summary.scalar('D_loss', D_loss)
        # debug
        self.fake_seq_img = fake_seq_img
        # train
        self.dis_train = dis_train
        self.G_train_op = G_train_op
        self.D_train_op = D_train_op
        self.iter_step = iter_step
        self.iter_step_op = iter_step_op
        # output
        self.fake_seq = fake_seq
        self.built = True

    def train(self, sample, fetch_data, continued=None):
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
            saver = tf.train.Saver()

            if continued:
                print 'restoring'
                saver.restore(sess, hparam.weight_path)
            else:
                sess.run(tf.global_variables_initializer())

            vis_code = sample(hparam.batch_size)
            history = sess.run(self.fake_seq_img,
                               feed_dict={self.code: vis_code})
            history = np.concatenate([history, history])

            i = begin = self.iter_step.eval()
            while i < hparam.iterations+begin:
                print i
                i += 1
                sess.run(self.iter_step_op)

                code = sample(hparam.batch_size)
                real_seq = fetch_data(hparam.batch_size)

                new_fake_seq_img = sess.run(self.fake_seq_img,
                                            feed_dict={self.code: code})
                replace_ind = np.random.choice(hparam.batch_size*2,
                                               size=(hparam.batch_size,),
                                               replace=False)
                history[replace_ind] = new_fake_seq_img
                train_ind = np.random.choice(hparam.batch_size*2,
                                             size=(hparam.batch_size,),
                                             replace=False)

                summary_D_loss, summary_real_img, _ = \
                    sess.run([self.summary_D_loss,
                              self.summary_real_img,
                              self.D_train_op],
                             feed_dict={self.real_seq: real_seq,
                                        self.fake_seq_img: history[train_ind],
                                        self.dis_train: True}
                             )

                if i < hparam.D_boost:
                    continue

                for _ in range(hparam.G_k):
                    summary_G_loss, _ = \
                        sess.run([self.summary_G_loss,
                                  self.G_train_op],
                                 feed_dict={self.code: code,
                                            self.dis_train: False})
                    # problematic with the reuse bn

                summary_fake_img = \
                    sess.run(self.summary_fake_img,
                             feed_dict={self.code: vis_code})

                train_writer.add_summary(summary_D_loss, i)
                train_writer.add_summary(summary_real_img, i)
                train_writer.add_summary(summary_G_loss, i)
                train_writer.add_summary(summary_fake_img, i)

                if i % 1000 == 0:
                    saver.save(sess, hparam.weight_path)


