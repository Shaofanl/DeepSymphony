import numpy as np
from keras.layers import Input, LSTM, Dense,\
        Activation, Dropout, LeakyReLU,\
        TimeDistributed
from keras.models import Model, load_model
from keras.optimizers import Adam
from midiwrapper import Song
from encoder_decoder import AllInOneEncoder
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint


def define_model(shape, stateful=False):
    if stateful:
        x = input = Input(batch_shape=shape)
    else:
        x = input = Input(shape)
    note_dim = shape[-1]
    x = LSTM(512,
             return_sequences=True,
             implementation=2,
             stateful=stateful)(x)
    x = LSTM(512,
             return_sequences=True,
             implementation=2,
             stateful=stateful)(x)
    x = LSTM(512,
             return_sequences=True,
             implementation=2,
             stateful=stateful)(x)
    x = TimeDistributed(Dense(note_dim))(x)
    x = TimeDistributed(Activation('softmax'))(x)
    model = Model(input, x)
    return model


if __name__ == '__main__':
    DIR = 'datasets/easymusicnotes/'
    # DIR 2 'datasets/TPD/jazz/'
    LEN = 2000  # length of input
    DIM = 128+128+100+7

    # preparing files
    # data = Song.load_from_dir("./datasets/easymusicnotes/",
    #                           encoder=AllInOneEncoder(return_indices=True))
    # data = Song.load_from_dir("./datasets/e-comp/",
    #                           encoder=AllInOneEncoder(return_indices=True))
    data = np.load('./datasets/e-comp-allinone.npz')['data']

    # sample training data
    def data_generator():
        batch_size = 64
        while True:
            x = []
            y = []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0]-LEN-1)
                x.append(data[ind][start:start+LEN])
                y.append(data[ind][start+1:start+LEN+1])
            x = np.array(x)
            y = np.array(y)

            # data argument
            # shift = np.random.choice([-3, +3, -4, +4])
            # x[x < 256] = np.clip(x[x < 256] + shift, 0, 256)
            # y[y < 256] = np.clip(y[y < 256] + shift, 0, 256)

            # sparse to complex
            x = to_categorical(x.flatten(), num_classes=DIM).\
                reshape(x.shape[0], x.shape[1], DIM)
            y = y.reshape(x.shape[0], x.shape[1], 1)
            yield (x, y)

    # Build models
    note_dim = DIM  # data[0].shape[-1]
    model = define_model((LEN, note_dim))

#   model = load_model('temp/simple_rnn.h5')
    model.load_weights('temp/simple_rnn.h5')
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(1e-5))

    checkpoint = ModelCheckpoint(filepath='temp/simple_rnn.h5',
                                 monitor='loss',
                                 verbose=1,
                                 save_best_only=True,)
    model.fit_generator(data_generator(),
                        steps_per_epoch=20,
                        epochs=500,
                        callbacks=[checkpoint])
    model.save("temp/simple_rnn.h5")
