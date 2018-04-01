import numpy as np
from keras.layers import LSTM, Input, \
    TimeDistributed, Dense, Embedding, Reshape, Concatenate
from keras.utils.np_utils import to_categorical
from keras.models import Model, load_model
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
from keras.optimizers import Adam, SGD
import music21 as ms


def handle(fn):
    return coder.encode(ms.converter.parse(fn))


if __name__ == '__main__':
    timesteps = 128
    batch_size = 32
    mode = 'train'
    mode = 'test'

    coder = NoteDurationCoder(normalize_key='C5',
                              resolution=1./16.)

    try:
        # raise Exception
        data = np.load('temp/piano-midi_duration.npz')
        notes = data['notes']
        durations = data['durations']
    except:
        data = np.array(map_dir(handle, './datasets/piano-midi.de/', cores=8))
        print map(lambda x: len(x[0]), data)
        data = filter(lambda x: len(x[0]) > 0, data)
        print map(lambda x: len(x[0]), data)
        notes, durations = zip(*data)
        np.savez('temp/piano-midi_duration.npz',
                 notes=notes, durations=durations)
    notes = np.array(notes)
    durations = np.array(durations)-1
    print 'durations'
    print np.min(map(lambda x: x.min(), durations))
    print np.max(map(lambda x: x.max(), durations))

    print 'notes'
    print np.min(map(lambda x: x.min(), notes))
    print np.max(map(lambda x: x.max(), notes))

    if mode == 'train':
        def fetch_data():
            while 1:
                x1, x2, y = [], [], []
                for _ in range(batch_size):
                    ind = np.random.randint(len(notes))
                    start = np.random.randint(notes[ind].shape[0]-timesteps-1)
                    notei = notes[ind][start:start+timesteps]
                    durai = durations[ind][start:start+timesteps]

                    diffi = np.random.randint(
                        -4, 4, size=durai.shape)
                    # diffi = diffi*np.random.uniform(0, 1.0)
                    diffi = np.round(diffi).astype('int')
                    wrong_durai = diffi+durai
                    wrong_durai = np.clip(wrong_durai, 0, 15)

                    x1.append(notei)
                    x2.append(wrong_durai)
                    y.append(durai)

                x1 = np.array(x1)
                x2 = np.array(x2)
                y = np.array(y)
                y = to_categorical(y, num_classes=16)
                yield [x1, x2], y
        (x1, x2), y = fetch_data().next()

        input1 = Input((None,))
        input2 = Input((None,))
        x1 = Embedding(129, 512)(input1)
        x2 = Embedding(16, 256)(input2)
        x1 = LSTM(64, return_sequences=True)(x1)
        x2 = LSTM(64, return_sequences=True)(x2)
        x = Concatenate(axis=-1)([x1, x2])
        x = LSTM(128, return_sequences=True)(x)
        x = TimeDistributed(Dense(200, activation='relu'))(x)
        x = TimeDistributed(Dense(16, activation='softmax'))(x)

        model = Model([input1, input2], x)
        model.summary()
        # model.compile('Adam', 'MSE')
        model.compile(Adam(5e-5),
                      'categorical_crossentropy',
                      ['categorical_accuracy'])
        model.load_weights('temp/duration.h5')
        model.fit_generator(fetch_data(),
                            steps_per_epoch=20,
                            epochs=10, verbose=1)
        model.save('temp/duration.h5')

    if mode == 'test':
        model = load_model('temp/duration.h5')
        model.summary()

        coder = NoteDurationCoder(resolution=1./16.)
        notes, duras = coder.encode(ms.converter.parse('example.mid'),
                                    force=True)
        # notes, duras = coder.encode(
        #     ms.converter.parse('datasets/piano-midi.de/schumann/schum_abegg.mid'),
        #     force=True)

        print notes
        print notes.shape
        duration = model.predict([np.array([notes]),
                                  np.array([duras])])[0].argmax(-1)+1
        print duration-duras
        print 'old', duras
        print 'new', duration
        print duration[-1]
        print 'sqr diff', ((duras-duration)**2).sum()
        coder.decode(notes, duration).write('midi', 'example.d.mid')
        # print zip(notes, duras)

    print 'done'
