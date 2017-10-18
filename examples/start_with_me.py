import numpy as np
from keras.utils.np_utils import to_categorical

from DeepSymphony.coders import AllInOneCoder
from DeepSymphony.models import StackedRNN
from DeepSymphony.utils import Song
from keras import optimizers


if __name__ == '__main__':
    LEN = 100  # length of input
    DIM = 128+128+100+7

    data = Song.load_from_dir("./datasets/easymusicnotes/",
                              encoder=AllInOneCoder(return_indices=True))

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
                       cells=[128, 128, 128],)
    model.build()
    model.model.load_weights('temp/start_with_me.h5')
#   model.train(data_generator(),
#               opt=optimizers.Adam(1e-3),
#               steps_per_epoch=30,
#               epochs=300,
#               save_path='temp/start_with_me.h5')

    model.build_generator('temp/start_with_me.h5')
    res = model.generate(seed=30, length=2000)

    mid = Song()
    track = mid.add_track()
    for msgi in AllInOneCoder().decode(res):
        track.append(msgi)
    mid.save_as('simple_rnn.mid')
