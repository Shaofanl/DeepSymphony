from keras.layers import Conv2D, Dense, Activation, \
    Input, BatchNormalization, Reshape, \
    UpSampling2D, Conv2DTranspose, LeakyReLU, \
    Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np

import os
from scipy.misc import imsave


def basic_gen(input_shape,
              img_shape,
              nf=128,
              scale=4,
              FC=[],
              use_upsample=False):
    h, w, dim = img_shape

    img = Input(input_shape)
    x = img
    for fc_dim in FC:
        x = Dense(fc_dim)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Dense(nf*2**(scale-1)*(h/2**scale)*(w/2**scale))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((h/2**scale, w/2**scale, nf*2**(scale-1)))(x)

    for s in range(scale-2, -1, -1):
        # up sample can elimiate the checkbroad artifact
        # http://distill.pub/2016/deconv-checkerboard/
        if use_upsample:
            x = UpSampling2D()(x)
            x = Conv2D(nf*2**s, (3, 3), padding='same')(x)
        else:
            x = Conv2DTranspose(nf*2**s,
                                (3, 3),
                                strides=(2, 2),
                                padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    if use_upsample:
        x = UpSampling2D()(x)
        x = Conv2D(dim, (3, 3), padding='same')(x)
    else:
        x = Conv2DTranspose(dim, (3, 3),
                            strides=(2, 2),
                            padding='same')(x)

    x = Activation('tanh')(x)

    return Model(img, x)


def basic_dis(input_shape, nf=128, scale=4, FC=[], bn=True):
    h, w, dim = input_shape

    img = Input(input_shape)
    x = img

    for s in range(scale):
        x = Conv2D(nf*2**s, (5, 5),
                   strides=(2, 2),
                   padding='same')(x)
        if bn:
            x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    for fc in FC:
        x = Dense(fc)(x)
        if bn:
            x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(img, x)


if __name__ == '__main__':
    coding = 200
    niter = 10000
    nbatch = 50
    vis_iterval = 100

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_train = x_train/127.5 - 1
    if not os.path.exists('temp/mnist_gan'):
        os.mkdir('temp/mnist_gan')

    def data_generator(bs):
        indices = np.random.randint(x_train.shape[0], size=(bs,))
        return x_train[indices]

    gen = basic_gen(input_shape=(coding,),
                    img_shape=x_train[0].shape,
                    nf=32,
                    scale=2,
                    FC=[64],)
    gen.summary()

    dis = basic_dis(input_shape=x_train[0].shape,
                    nf=32,
                    scale=2,
                    FC=[64],)
    dis.summary()

    opt = Adam(1e-3, beta_1=0.5, beta_2=0.9)

    gendis = Sequential([gen, dis])
    dis.trainable = False
    gendis.compile(optimizer=opt, loss='binary_crossentropy')

    shape = dis.get_input_shape_at(0)[1:]
    gen_input, real_input = Input(shape), Input(shape)
    dis2batch = Model([gen_input, real_input],
                      [dis(gen_input), dis(real_input)])
    dis.trainable = True
    dis2batch.compile(optimizer=opt,
                      loss='binary_crossentropy',
                      metrics=['binary_accuracy'])

    gen_trainner = gendis
    dis_trainner = dis2batch

    vis_Z = np.random.uniform(-1., 1., size=(25, coding)).astype('float32')
    for iteration in range(1, niter+1):
        print 'iteration', iteration
        real_img = data_generator(nbatch)
        Z = np.random.uniform(-1., 1., size=(nbatch, coding)).astype('float32')
        gen_img = gen.predict(Z)

        y = np.ones((nbatch, 1))
        g_loss = gen_trainner.train_on_batch(Z, y)

        gen_y = np.zeros((nbatch, 1))
        real_y = np.ones((nbatch, 1))
        d_loss = dis_trainner.train_on_batch([gen_img, real_img],
                                             [gen_y, real_y])

        if iteration % vis_iterval == 0:
            vis_img = gen.predict(vis_Z).reshape(5, 5, 28, 28)
            vis_big = np.ones((5*28, 5*28))
            for i in range(5):
                for j in range(5):
                    vis_big[i*28:(i+1)*28, j*28:(j+1)*28] = \
                            ((vis_img[i, j]+1)*127.5).astype('int')
            imsave('temp/mnist_gan/{:03d}.png'.format(iteration),
                   vis_big)
