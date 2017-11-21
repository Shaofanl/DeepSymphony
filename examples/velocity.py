import numpy as np
from keras.layers import LSTM, Input, TimeDistributed, Dense
from keras.models import Model, load_model
from DeepSymphony.utils.Music21Coder import MultiHotCoder
import music21 as ms

if __name__ == '__main__':
    timesteps = 128
    batch_size = 30

    mode = 'train'
    mode = 'test'
    if mode == 'train':
        data = np.load('temp/piano-midi.npz')
        voi, vel = data['voice'], data['velocity']
        voi = filter(lambda x: len(x) > 0 and x.shape[0] > timesteps, voi)
        vel = filter(lambda x: len(x) > 0 and x.shape[0] > timesteps, vel)

        def fetch_data():
            while 1:
                x, y = [], []
                for _ in range(batch_size):
                    ind = np.random.randint(len(voi))
                    start = np.random.randint(voi[ind].shape[0]-timesteps-1)
                    voii = voi[ind][start:start+timesteps]
                    veli = vel[ind][start:start+timesteps]
                    x.append(voii)
                    y.append(veli)

                x = np.array(x)
                y = np.array(y)/127.*2-1
                yield x, y
        x, y = fetch_data().next()

        input = x = Input((None, 128))
        x = LSTM(64, return_sequences=True)(x)
        x = LSTM(64, return_sequences=True)(x)
        x = TimeDistributed(Dense(128, activation='tanh'))(x)

        model = Model(input, x)
        model.summary()
        model.compile('Adam', 'MSE')
        model.load_weights('temp/velocity.h5')
        model.fit_generator(fetch_data(),
                            steps_per_epoch=200,
                            epochs=10, verbose=1)
        model.save('temp/velocity.h5')

    if mode == 'test':
        model = load_model('temp/velocity.h5')
        model.summary()

        coder = MultiHotCoder()
        notes = coder.encode(ms.converter.parse('example.mid'), force=True)
        velocity = model.predict(np.array([notes]))[0]
        velocity = ((velocity+1)/2.*127.).astype('uint8')

        coder.decode(notes, velocity).write('midi', 'example.v.mid')
        print (velocity*notes).max(), (velocity*notes).min()

        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.imshow(notes.T[::-1, :])
        plt.subplot(212)
        plt.imshow(velocity.T[::-1, :])
        plt.show()

    print 'done'
