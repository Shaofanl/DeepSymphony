import numpy as np
from keras.utils.np_utils import to_categorical

from DS.coders import AllInOneCoder
from DS.models import StackedRNN
from DS.utils import Song


if __name__ == '__main__':
    LEN = 2000  # length of input
    DIM = 128+128+100+7

    data = np.load('./datasets/e-comp-allinone.npz')['data']

    # sample training data
    def data_generator():
        batch_size = 32
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

            # sparse to complex
            x = to_categorical(x.flatten(), num_classes=DIM).\
                reshape(x.shape[0], x.shape[1], DIM)
            y = y.reshape(x.shape[0], x.shape[1], 1)
            yield (x, y)

    # model
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM,
                       output_dim=DIM,
                       cells=[512, 512, 512])
    model.build()
    model.train(data_generator(),
                steps_per_epoch=5,
                epochs=1,
                save_path='/tmp/tmp.h5')

    res = model.generate('/tmp/tmp.h5')

    mid = Song()
    track = mid.add_track()
    for msgi in AllInOneCoder().decode(res):
        track.append(msgi)
    mid.save_as('simple_rnn.mid')
