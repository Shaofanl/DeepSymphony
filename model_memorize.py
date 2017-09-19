from utils import getAbsT, getHots
from mido import MidiFile
import os
import numpy as np

import keras
from keras.layers import Input, LSTM, Dense, Activation
from keras.models import Model
from keras.optimizers import RMSprop, Adam


if __name__ == '__main__':
    FILE = 'datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid'
    LEN = 20
    dim = 128

    midi = MidiFile(FILE)
    msgs, times = getAbsT(midi, filter_f=lambda x: x.type in ['note_on', 'note_off'], unit='beat')
    hots = getHots(msgs, times, resolution=0.25)
    print hots.shape

    x = []
    y = []
    for i in range(LEN, len(hots)-LEN-1):
        x.append(hots[i:i+LEN])
        y.append(hots[i+LEN])
    train_x = np.array(x)
    train_y = np.array(y)

    print train_x.shape
    print train_y.shape

    #Build models
    x = input = Input(batch_shape=(1, LEN, dim))
    x = LSTM(100, stateful=True, )(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(128, activation='sigmoid')(x)
    model = Model(input, x)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(1e-3), metrics=[])

    model.fit(train_x, train_y, epochs=40, batch_size=1)
    model.save("temp/memorize.h5")


