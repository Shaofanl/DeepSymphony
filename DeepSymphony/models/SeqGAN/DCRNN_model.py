# sequential GAN with Deconvolutional RNN
#  RNN1              RNN1
#   o1    o1    o1    o2    o2   o2
#  RNN2  RNN2  RNN2  RNN2  RNN2  RNN2
#   o1    o2    o3    o4    o5   o6

from DeepSymphony.common.HParam import HParam
from DeepSymphony.common.resnet import ResNetBuilder

import os
import shutil
import numpy as np

import tensorflow as tf
from tensorflow.contrib import rnn, slim, layers


def safe_log(x):
    epsilon = 1e-6
    return tf.log(tf.clip_by_value(x, epsilon, 1-epsilon))


def lrelu(x):
    return tf.maximum(0.2*x, x)


class DCRNNHParam(HParam):
    # basic
    cells = [256, 128, 128]
    repeats = [8, 2, 1]
    basic_cell = rnn.GRUCell
    embed_dim = 500
    code_dim = 200
    trainable_gen = ['generator']
    last_bidirectional = False
    # training
    D_lr = 1e-5
    G_lr = 1e-5
    batch_size = 32
    iterations = 300
    # logs
    workdir = './temp/DCRNN/'
    tensorboard_dir = 'tensorboard/'
    overwrite_workdir = False
    # debug
    debug = False

    def __init__(self, **kwargs):
        self.register_check('timesteps')
        self.register_check('vocab_size')
        self.register_check('onehot')

        super(DCRNNHParam, self).__init__(**kwargs)


class DCRNN(object):
    def __init__(self, hparam):
        self.built = False
        if not hasattr(hparam, 'weight_path'):
            hparam.weight_path = \
                os.path.join(hparam.workdir, 'DCRNN.ckpt')
        self.hparam = hparam
        # self.start_token = hparam.vocab_size
        # self.eos_token = hparam.vocab_size+1

    def build(self):
        hparam = self.hparam

        if hparam.linspace_code:
            code = tf.placeholder("float32", [None,
                                              hparam.timesteps,
                                              hparam.code_dim],
                                  name='code')
        else:
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
            # play with code

            step = int(hparam.timesteps / np.prod(hparam.repeats))
            first_input = code if hparam.linspace_code else \
                tf.tile(tf.expand_dims(code, 1), (1, step, 1))
            if hparam.timestep_pad:
                first_input = tf.concat(
                    [
                        first_input,
                        tf.tile(tf.expand_dims(tf.expand_dims(tf.lin_space(
                            0., 1., step), 0), -1),
                            (bs, 1, 1)),
                    ], -1
                )
            outputs = [first_input]
            ind = 0
            for repeat, cell_size in zip(hparam.repeats, hparam.cells):
                ind += 1
                with tf.variable_scope('layer{}'.format(ind)):
                    if ind == len(hparam.repeats) and \
                       hparam.last_bidirectional:
                        assert(repeat == 1)
                        fw_cell = hparam.basic_cell(cell_size)
                        bw_cell = hparam.basic_cell(cell_size)
                        output, state = tf.nn.bidirectional_dynamic_rnn(
                            fw_cell, bw_cell,
                            outputs[-1],
                            dtype=tf.float32,
                        )
                        output = tf.concat(output, 2)
                        cell_size *= 2
                    else:
                        cell = hparam.basic_cell(cell_size)
                        output, state = tf.nn.dynamic_rnn(
                            cell,
                            outputs[-1],
                            dtype=tf.float32,
                        )
                    # output = output * 2
                    if repeat != 1:
                        step *= repeat
                        output = tf.reshape(
                            tf.tile(output, (1, 1, repeat)),
                            [bs, step, cell_size])
                    outputs.append(output)
            outputs[-1] = outputs[-1][:, :hparam.timesteps, :]
            for o in outputs:
                print o

            with tf.variable_scope('decision'):
                fake_seq_img = outputs[-1]
                fake_seq_img = layers.linear(fake_seq_img, hparam.vocab_size)
                outputs.append(fake_seq_img)
                fake_seq_img = tf.tanh(fake_seq_img)
                # fake_seq_img = tf.nn.softmax(fake_seq_img, -1)
                outputs.append(fake_seq_img)
                fake_seq = tf.argmax(fake_seq_img, -1)

        # discriminator
        def dis(seq_img, bn_scope, reuse=False):
            with tf.variable_scope('discriminator', reuse=reuse):
                # x = tf.nn.embedding_lookup(embeddings, seq)
                x = tf.expand_dims(seq_img, -1)
                # x = ResNetBuilder(dis_train,
                #                   bn_scopes=['fake', 'real'],
                #                   bn_scope=bn_scope).\
                #     resnet(x, structure=[2, 2, 2, 2], filters=8, nb_class=1)

                #  note axis
                fs = 32
                x = lrelu(slim.conv2d(x, fs*1, [5, 5], stride=2))
                x = lrelu(slim.conv2d(x, fs*2, [5, 5], stride=2))
                x = lrelu(slim.conv2d(x, fs*4, [5, 5], stride=2))
                x = lrelu(slim.conv2d(x, fs*4, [5, 5], stride=2))
                x = slim.flatten(x)
                x = slim.linear(x, 1)
                # x = tf.nn.sigmoid(x)
            return x
        # opt
        # problematic with the reuse bn
        # fake_seq_img = tf.where(
        #     tf.greater(fake_seq_img, 0.5),
        #     fake_seq_img,
        #     tf.zeros_like(fake_seq_img))
        fake_dis_pred = dis(fake_seq_img, bn_scope='fake')
        real_dis_pred = dis(real_seq_img, bn_scope='real', reuse=True)

        # traditional GAN loss
        # G_loss = tf.reduce_mean(-safe_log(fake_dis_pred))
        # D_loss = tf.reduce_mean(-safe_log(real_dis_pred)) +\
        #     tf.reduce_mean(-safe_log(1-fake_dis_pred))
        # IWGAN
        epsilon = tf.random_uniform(
            minval=0, maxval=1.0,
            shape=[tf.shape(real_seq_img)[0],
                   tf.shape(real_seq_img)[1]])
        epsilon = tf.tile(tf.expand_dims(epsilon, -1),
                          (1, 1, tf.shape(real_seq_img)[2]))
        intepolation = fake_seq_img*epsilon+real_seq_img*(1.0-epsilon)
        inte_dis_pred = dis(intepolation, bn_scope='intepolation', reuse=True)
        grad = tf.gradients(inte_dis_pred, intepolation)[0]
        grad = tf.reshape(grad, (-1, hparam.timesteps*hparam.vocab_size))
        D_loss = tf.reduce_mean(fake_dis_pred) - \
            tf.reduce_mean(real_dis_pred) + \
            10*tf.reduce_mean(tf.square(tf.norm(grad, ord=2, axis=1)-1))
        G_loss = -tf.reduce_mean(fake_dis_pred)

        fake_seq_img_grad = tf.gradients(G_loss, fake_seq_img)[0]

        G_opt = tf.train.AdamOptimizer(learning_rate=hparam.G_lr,
                                       beta1=0.5, beta2=0.9)
        # D_opt = tf.train.GradientDescentOptimizer(learning_rate=hparam.D_lr)
        D_opt = tf.train.AdamOptimizer(learning_rate=hparam.D_lr,
                                       beta1=0.5, beta2=0.9)
        D_iter = tf.Variable(0, name='D_iter')
        G_iter = tf.Variable(0, name='G_iter')
        trainable_gen_var = reduce(
            lambda x, y: x+y,
            [tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, ele)
                for ele in hparam.trainable_gen],
            []
        )
        G_train_op = slim.learning.create_train_op(
            G_loss, G_opt,
            variables_to_train=trainable_gen_var,
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
        self.real_seq_img = real_seq_img
        # summary
        self.summary_fake_img = tf.summary.image(
            'fake_img', tf.expand_dims(fake_seq_img, -1))
        self.summary_real_img = tf.summary.image(
            'real_img', tf.expand_dims(real_seq_img, -1))
        self.summary_G_loss = tf.summary.scalar('G_loss', G_loss)
        self.summary_D_loss = tf.summary.scalar('D_loss', D_loss)
        self.summary_fake_dis_pred = tf.summary.scalar(
            'fake_dis_pred', tf.reduce_mean(fake_dis_pred))
        self.summary_real_dis_pred = tf.summary.scalar(
            'real_dis_pred', tf.reduce_mean(real_dis_pred))
        self.summary_fake_img_grad = tf.summary.image(
            'gradient_map', tf.expand_dims(fake_seq_img_grad, -1))
        self.gen_outputs = outputs

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

    @property
    def major_params(self):
        return \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'generator') +\
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'discriminator') +\
                [self.iter_step]

    @property
    def gen_params(self):
        return \
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'generator')

    def generate(self, code, img=False):
        hparam = self.hparam
        with tf.Session() as sess:
            saver = tf.train.Saver(self.gen_params)
            saver.restore(sess, hparam.weight_path)

            tensor = self.fake_seq_img if img else self.fake_seq

            seq = sess.run(tensor,
                           feed_dict={self.code: code})
            return seq

    def analyze(self, sample):
        hparam = self.hparam
        with tf.Session() as sess:
            saver = tf.train.Saver(self.major_params)
            saver.restore(sess, hparam.weight_path)

            code = sample(hparam.batch_size)

            outputs = sess.run(self.gen_outputs,
                               feed_dict={self.code: code})
            outputs[0] = outputs[0][:, :, :-1]

            print np.var(code, 0).mean()
            for o in outputs:
                o = o.reshape(o.shape[0], -1)
                print np.var(o, 0).mean()

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
            saver = tf.train.Saver(
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'generator') +
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                  'discriminator') +
                [self.iter_step]
            )

            sess.run(tf.global_variables_initializer())
            if continued:
                print 'restoring'
                saver.restore(sess, hparam.weight_path)

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
                real_seq_tensor = self.real_seq if hparam.onehot else\
                    self.real_seq_img

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
                             feed_dict={real_seq_tensor: real_seq,
                                        self.fake_seq_img: history[train_ind],
                                        self.dis_train: True}
                             )
                train_writer.add_summary(summary_real_img, i)
                train_writer.add_summary(summary_D_loss, i)

                summary_fake_dis_pred, summary_real_dis_pred = \
                    sess.run([self.summary_fake_dis_pred,
                              self.summary_real_dis_pred],
                             feed_dict={real_seq_tensor: real_seq,
                                        self.fake_seq_img: history[train_ind],
                                        self.dis_train: False})
                train_writer.add_summary(summary_fake_dis_pred, i)
                train_writer.add_summary(summary_real_dis_pred, i)

                summary_fake_img, summary_fake_img_grad = \
                    sess.run([self.summary_fake_img,
                              self.summary_fake_img_grad],
                             feed_dict={self.code: vis_code})

                train_writer.add_summary(summary_fake_img, i)
                train_writer.add_summary(summary_fake_img_grad, i)

                if i < hparam.D_boost:
                    continue

                for _ in range(hparam.G_k):
                    summary_G_loss, _ = \
                        sess.run([self.summary_G_loss,
                                  self.G_train_op],
                                 feed_dict={self.code: code,
                                            self.dis_train: False})
                    # problematic with the reuse bn
                train_writer.add_summary(summary_G_loss, i)

                if i % 100 == 0:
                    saver.save(sess, hparam.weight_path)
            saver.save(sess, hparam.weight_path)
