import numpy as np
from keras.layers import Embedding

from DeepSymphony.coders import AllInOneCoder
from DeepSymphony.models import StackedRNN
from DeepSymphony.utils import Song
from keras.models import load_model

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
            y = y.reshape(x.shape[0], x.shape[1], 1)
            yield (x, y)

    # model
    emb_w = load_model('temp/emb.h5').layers[2].layers[1].get_weights()
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM,
                       output_dim=DIM,
                       cells=[512, 512, 512],
                       embedding=Embedding(DIM, 512, trainable=False),
                       embedding_w=emb_w)
    model.build()
    model.model.load_weights('temp/emb_stackrnn.h5')

    model.train(data_generator(),
                opt=1e-5,
                steps_per_epoch=20,
                epochs=200,
                save_path='temp/emb_stackrnn.h5')

    res = model.generate('temp/emb_stackrnn.h5',
                         seed=64,
                         length=5000)
    mid = Song()
    track = mid.add_track()
    for msgi in AllInOneCoder().decode(res):
        track.append(msgi)
    mid.save_as('simple_rnn.mid')
