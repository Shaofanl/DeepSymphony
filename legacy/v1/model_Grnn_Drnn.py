'''
    note1   note2   note3   note4
     ^       ^       ^       ^
    RNN --> RNN --> RNN --> RNN
     ^       ^       ^       ^
    code     0       0       0
'''
import os
import numpy as np
from scipy.misc import imsave
from midiwrapper import Song
import shutil

from keras.layers import Conv2D, Dense, Activation, \
    Input, BatchNormalization, Reshape, \
    UpSampling2D, Conv2DTranspose, LeakyReLU, \
    Flatten, LSTM, Dropout, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from encoder_decoder import AllInOneEncoder, OneHotEncoder
import keras.backend as K
from keras.utils.np_utils import to_categorical

import argparse
parser = argparse.ArgumentParser(description='GAN-RNN Model')
parser.add_argument('--activation', type=str, default='sigmoid',
                    help='define activation and normalization range')
parser.add_argument('--feed', type=str, default='first',
                    help='feed code only in the first place or all]')

parser.add_argument('--code_dim', type=int, default=200)
# parser.add_argument('--note_dim', type=int, default=128)
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--nbatch', type=int, default=32)
parser.add_argument('--niter', type=int, default=1000)

# training policy
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--boost', type=int, default=10)

parser.add_argument('--vis_interval', type=int, default=20)
parser.add_argument('--work_dir', type=str, default='temp/gan',
                    help='work dir')

args = parser.parse_args()
print args


def rnn_gen(coding_shape,
            seq_shape,
            lstm=[128],
            fc=[]):
    """
        coding should be in (bs, len, code_dim)
    """
    x = code = Input(coding_shape)

    for lstm_i in lstm:
        x = LSTM(lstm_i, return_sequences=True, dropout=0.5)(x)

    if len(fc) > 0:
        fcx = fc_input = Input(K.get_variable_shape(x)[2:])
        for ind, fc_i in enumerate(fc):
            fcx = Dense(fc_i, )(fcx)
            if ind == len(fc)-1:
                fcx = Activation(args.activation)(fcx)
                fcx = Dropout(0.5)(fcx)
            else:
                fcx = LeakyReLU(0.2)(fcx)
        fc_model = Model(fc_input, fcx)
        fc_model.summary()
        x = TimeDistributed(fc_model)(x)
    else:
        x = Activation(args.activation)(x)
    return Model(code, x)


def rnn_dis(seq_shape, lstm=[128], fc=[10]):
    x = seq = Input(seq_shape)

    for lstm_i in lstm[:-1]:
        x = LSTM(lstm_i, return_sequences=True,)(x)
    x = LSTM(lstm[-1])(x)
    for fc_i in fc:
        x = Dense(fc_i)(x)
        x = LeakyReLU(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(seq, x)


if __name__ == '__main__':
    seq_len = args.seq_len
    code_dim = args.code_dim
    # note_dim = args.note_dim
    niter = args.niter
    nbatch = args.nbatch
    vis_iterval = args.vis_interval
    work_dir = args.work_dir
    vis_dir = os.path.join(args.work_dir, 'vis')

    # env setup
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.mkdir(work_dir)
    os.mkdir(vis_dir)

    # data preparation
    data = Song.load_from_dir("../../datasets/easymusicnotes/",
                              encoder=OneHotEncoder())
# data = Song.load_from_dir("./datasets/e-comp/", encoder=AllInOneEncoder())
    # data = np.load('../../datasets/e-comp-allinone.npz')['data']
    note_dim = data[0].shape[-1]
    print min(map(lambda x: x.shape[0], data))

    def data_generator(bs):
        indices = np.random.randint(data.shape[0], size=(bs,))
        x = []
        for ind_i in indices:
            start = np.random.randint(data[ind_i].shape[0]-seq_len)
            x.append(data[ind_i][start:start+seq_len])
        x = np.array(x).astype('float')
        if args.activation == 'tanh':
            x = x*2 - 1
        return x

    def code_generator(bs):
        Z = np.random.uniform(-1., 1.,
                              size=(bs, code_dim))
        Z_pad = np.zeros((bs, seq_len, code_dim))
        if args.feed == 'first':
            Z_pad[:, 0, :] = Z
        if args.feed == 'all':
            Z_pad = Z[:, None, :]
            import ipdb
            ipdb.set_trace()
        return Z_pad

    # component definition
    gen = rnn_gen(coding_shape=(seq_len, code_dim),
                  seq_shape=(seq_len, note_dim),
                  lstm=[512, 512, note_dim],
                  fc=[])  # note_dim])
    gen.summary()

    dis = rnn_dis(seq_shape=(seq_len, note_dim),
                  lstm=[512, 512],
                  fc=[])
    dis.summary()

    # model definition
    gen_opt = Adam(5e-4, beta_1=0.5, beta_2=0.9)
    dis_opt = SGD(5e-4)  # , momentum=0.9)

    gendis = Sequential([gen, dis])
    dis.trainable = False
    gendis.compile(optimizer=gen_opt, loss='binary_crossentropy')

    shape = dis.get_input_shape_at(0)[1:]
    gen_input, real_input = Input(shape), Input(shape)
    dis2batch = Model([gen_input, real_input],
                      [dis(gen_input), dis(real_input)])
    dis.trainable = True
    dis2batch.compile(optimizer=dis_opt,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

    gen_trainner = gendis
    dis_trainner = dis2batch

    imsave('{}/real.png'.format(vis_dir),
           Song.grid_vis_songs(data_generator(25)))
    vis_Z = code_generator(25)
    for iteration in range(0, niter):
        print 'iteration', iteration
        Z = code_generator(nbatch)
        gen_img = gen.predict(Z)

        if iteration < args.boost or iteration % args.k == 0:
            real_img = data_generator(nbatch)
            gen_y = np.zeros((nbatch, 1))
            real_y = np.ones((nbatch, 1))
            d_loss = dis_trainner.train_on_batch([gen_img, real_img],
                                                 [gen_y, real_y])
            print('\tDiscriminator:\t{}'.format(d_loss))
        else:
            y = np.ones((nbatch, 1))
            g_loss = gen_trainner.train_on_batch(Z, y)
            print('\tGenerator:\t{}'.format(g_loss))

        if iteration % vis_iterval == 0:
            songs = gen.predict(vis_Z)
            imsave('{}/{:03d}.png'.format(vis_dir, iteration),
                   Song.grid_vis_songs(songs))
    gen.save('{}/gen.h5'.format(work_dir))
