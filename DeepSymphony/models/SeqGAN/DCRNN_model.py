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
from types import GeneratorType

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
    bidirection = False
    plus_code = False
    show_grad = False
    show_input = False
    rnn_dis = False
    deconv_decision = False
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

        if hparam.code_ndim == 3:
            code = tf.placeholder("float32", [None,
                                              hparam.timesteps,
                                              hparam.code_dim],
                                  name='code')
        elif hparam.code_ndim == 2:
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
        final_states = []
        init_states = []
        with tf.variable_scope('generator'):
            # play with code

            step = int(hparam.timesteps / np.prod(hparam.repeats))
            first_input = code if hparam.code_ndim == 3 else \
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
            for ind in range(len(hparam.cells)):
                repeat = hparam.repeats[ind]
                cell_size = hparam.cells[ind]
                bi = hparam.bidirection[ind]
                with tf.variable_scope('layer{}'.format(ind)):
                    # if ind == len(hparam.repeats) and \
                    #    hparam.last_bidirectional:
                    if bi:
                        # assert(repeat == 1)
                        fw_cell = hparam.basic_cell(cell_size)
                        bw_cell = hparam.basic_cell(cell_size)
                        fw_init = fw_cell.zero_state(bs, tf.float32)
                        bw_init = fw_cell.zero_state(bs, tf.float32)
                        output, state = tf.nn.bidirectional_dynamic_rnn(
                            fw_cell, bw_cell,
                            outputs[-1],
                            initial_state_fw=fw_init,
                            initial_state_bw=bw_init,
                            dtype=tf.float32,
                        )
                        output = tf.concat(output, 2)
                        cell_size *= 2

                        init_states.extend([fw_init, bw_init])
                        final_states.append(state)
                    else:
                        cell = hparam.basic_cell(cell_size)
                        init = cell.zero_state(bs, tf.float32)
                        output, state = tf.nn.dynamic_rnn(
                            cell,
                            outputs[-1],
                            dtype=tf.float32,
                        )
                        init_states.append(init)
                        final_states.append(state)
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
                if hparam.deconv_decision:
                    fake_seq_img = outputs[-1]

                    right = slim.fully_connected(
                        fake_seq_img, 32*10, activation_fn=None)
                    right = tf.reshape(
                        right, [bs, hparam.timesteps, 10, 32])
                    # right = tf.nn.softmax(right, -1)
                    right = slim.conv2d_transpose(
                        right, 1, [1, 12], stride=(1, 12),
                        activation_fn=tf.tanh)[:, :, :, 0]

                    fake_seq_img = right
                    fake_seq_img = tf.concat(
                        [tf.ones([bs, hparam.timesteps, 4])*-1,
                         fake_seq_img,
                         tf.ones([bs, hparam.timesteps, 4])*-1],
                        axis=2)
                    print fake_seq_img
                else:
                    fake_seq_img = outputs[-1]
                    fake_seq_img = layers.linear(fake_seq_img, hparam.vocab_size)
                    outputs.append(fake_seq_img)
                    fake_seq_img = tf.tanh(fake_seq_img)
                    # fake_seq_img = tf.nn.softmax(fake_seq_img, -1)
                    outputs.append(fake_seq_img)
                fake_seq = tf.argmax(fake_seq_img, -1)

            if hparam.plus_code:
                fake_seq_img = tf.clip_by_value(fake_seq_img+code, -1., +1.)

        # discriminator
        if hparam.rnn_dis:
            def dis(seq_img, bn_scope, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    print 'dis'
                    slices = tf.unstack(seq_img, axis=1)
                    fw_cell = hparam.basic_cell(32)
                    # bw_cell = hparam.basic_cell(64)
                    x, state = tf.nn.static_rnn(
                        fw_cell,  # bw_cell,
                        slices,
                        dtype=tf.float32,
                    )
                    x = tf.stack(x, axis=1)
                    print x
                    # x = tf.concat(x, 2)
                    x = slim.linear(x, 1)
                    print x
                    x = slim.flatten(x)
                    print x
                    x = slim.linear(x, 1)
                    print x
                    # x = tf.nn.sigmoid(x)
                return x
        else:
            def dis(seq_img, bn_scope, reuse=False):
                with tf.variable_scope('discriminator', reuse=reuse):
                    fs = 32
                    covariance = tf.matmul(seq_img, seq_img, transpose_b=True)
                    x = tf.expand_dims(covariance, -1)
                    x = lrelu(slim.conv2d(x, fs*1, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*2, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*4, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*4, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    covariance_feat = slim.flatten(x)

                    # x = tf.nn.embedding_lookup(embeddings, seq)
                    # x = ResNetBuilder(dis_train,
                    #                   bn_scopes=['fake', 'real'],
                    #                   bn_scope=bn_scope).\
                    #     resnet(x, structure=[2, 2, 2, 2], filters=8, nb_class=1)

                    #  note axis
                    fs = 32
                    x = seq_img
                    x = tf.expand_dims(seq_img, -1)
                    x = lrelu(slim.conv2d(x, fs*1, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*2, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*4, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    x = lrelu(slim.conv2d(x, fs*4, [5, 5]))
                    x = slim.max_pool2d(x, (2, 2))
                    seq_feat = slim.flatten(x)

                    feat = tf.concat([covariance_feat, seq_feat], axis=1)

                    x = slim.linear(feat, 1)
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
        print 'grad'
        intepolation = fake_seq_img*epsilon+real_seq_img*(1.0-epsilon)
        inte_dis_pred = dis(intepolation, bn_scope='intepolation', reuse=True)
        grad = tf.gradients(inte_dis_pred, intepolation)[0]
        print grad
        grad = tf.reshape(grad, (-1, hparam.timesteps*hparam.vocab_size))
        print grad
        D_loss = tf.reduce_mean(fake_dis_pred) - \
            tf.reduce_mean(real_dis_pred) + \
            10*tf.reduce_mean(tf.square(tf.norm(grad, ord=2, axis=1)-1))
        G_loss = -tf.reduce_mean(fake_dis_pred)
        print D_loss
        print G_loss

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
        self.summary_first_input = tf.summary.image(
            'noise', tf.expand_dims(first_input, -1))
        self.gen_outputs = outputs

        # debug
        self.fake_seq_img = fake_seq_img
        self.first_input = first_input
        self.init_states = tuple(init_states)
        self.final_states = tuple(final_states)
        self.bs_tensor = bs
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

    def generate(self, code,
                 real_time_callback=None,
                 code_img=False, img=False):
        hparam = self.hparam
        with tf.Session() as sess:
            saver = tf.train.Saver(self.gen_params+[self.iter_step])
            saver.restore(sess, hparam.weight_path)
            print 'restore from iteration', self.iter_step.eval()

            input_tensor = self.first_input if code_img else self.code
            tensor = self.fake_seq_img if img else self.fake_seq

            if real_time_callback is not None:
                state = sess.run(self.init_states,
                                 feed_dict={self.bs_tensor: 1})
                for codei in code:
                    seq, state = sess.run(
                        [tensor, self.final_states],
                        feed_dict={input_tensor: codei,
                                   self.init_states: state,
                                   self.bs_tensor: len(codei)})
                    real_time_callback(seq)
            else:
                seq = sess.run(tensor, feed_dict={input_tensor: code,
                                                  self.bs_tensor: len(code)})
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

        gpu_opt = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt)) as sess:
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

                if hparam.show_grad:
                    summary_fake_img, summary_fake_img_grad = \
                        sess.run([self.summary_fake_img,
                                  self.summary_fake_img_grad],
                                 feed_dict={self.code: vis_code})
                    train_writer.add_summary(summary_fake_img, i)
                    train_writer.add_summary(summary_fake_img_grad, i)
                else:
                    summary_fake_img = \
                        sess.run(self.summary_fake_img,
                                 feed_dict={self.code: vis_code})
                    train_writer.add_summary(summary_fake_img, i)

                if hparam.show_input:
                    summary_first_input = \
                        sess.run(self.summary_first_input,
                                 feed_dict={self.code: vis_code})
                    train_writer.add_summary(summary_first_input, i)

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
