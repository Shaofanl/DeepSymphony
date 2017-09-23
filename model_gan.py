from midiwrapper import Song
import os
import numpy as np

from keras.layers import Input, LSTM, Dense, Dropout
from keras.models import Model
from keras.optimizers import RMSprop


if __name__ == '__main__':
    DIR = 'datasets/easymusicnotes/'
    LEN = 200  # length of input
    N = 10000  # number of training sequences

    # preparing files
    print 'Reading files ...'
    filelist = []
    for root, _, files in os.walk(DIR):
        for name in files:
            filelist.append(os.path.join(root, name))
    midis = [Song(filename) for filename in filelist]
    data = []
    for ind, midi in enumerate(midis):
        print '\t[{:02d}/{:02d}] Handling'.format(ind, len(midis)),\
            filelist[ind], '...'
        hots = midi.encode_onehot(
              {'filter_f': lambda x: x.type in ['note_on', 'note_off'],
                  'unit': 'beat'},
              {'resolution': 0.25})
        data.append(hots)
        print '\t', hots.shape
    data = np.array(data)

    # sample training data
    print 'Sampling ...'
    x = []
    y = []
    for _ in range(N):
        ind = np.random.randint(len(data))
        start = np.random.randint(data[ind].shape[0]-LEN-1)
        x.append(data[ind][start: start+LEN])
        y.append(data[ind][start+LEN])

    x = np.array(x)  # (N, LEN, dim)
    y = np.array(y)  # (N, dim)
    dim = x.shape[-1]
    print '\tx.shape =', x.shape, 'y.shape =', y.shape
#   np.savez("temp/data.npz", x=x, y=y)
#   print '\tsaved to temp/data.npz'

    train_x = x[:int(N*0.8)]
    train_y = y[:int(N*0.8)]
    valid_x = x[int(N*0.8):]
    valid_y = y[int(N*0.8):]

    # Build models
    x = input = Input((LEN, dim))
#   x = input = Input(batch_shape=(1,LEN, dim),)
    x = LSTM(512, stateful=False, return_sequences=True)(x)
    x = LSTM(512, stateful=False, return_sequences=True)(x)
    x = LSTM(512, stateful=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='sigmoid')(x)
    model = Model(input, x)
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(1e-3),
                  metrics=[])

    model.fit(train_x, train_y,
              epochs=200,
              validation_data=(valid_x, valid_y),
              batch_size=32)
    # batch_size=1)
    model.save("temp/simple_rnn.h5")
