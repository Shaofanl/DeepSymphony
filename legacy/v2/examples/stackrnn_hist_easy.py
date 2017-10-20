import numpy as np
from keras.utils.np_utils import to_categorical

from DeepSymphony.coders import AllInOneCoder
from DeepSymphony.models import StackedRNN
from DeepSymphony.utils import Song
from DeepSymphony.utils.constants import NOTE_NUMBER
from DeepSymphony.utils.stat import histogram, histogram_onehot,\
    min_norm, norm


if __name__ == '__main__':
    LEN = 100  # length of input
    DIM_IN = 128+128+100+7+len(NOTE_NUMBER)
    DIM_OUT = 128+128+100+7
    coder = AllInOneCoder()
    H_TEMP = 0.05

    data, filelist = Song.load_from_dir(
        "./datasets/easymusicnotes/",
        encoder=AllInOneCoder(return_indices=True),
        return_list=True)
    hist = np.load('./datasets/e-comp-allinone-hist.npz')['hist']

    def data_generator():
        batch_size = 32
        while True:
            x = []
            y = []
            h = []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0]-LEN-1)
                x.append(data[ind][start:start+LEN])
                y.append(data[ind][start+1:start+LEN+1])
                # global hist
                # h.append(hist[ind])
                # patch hist
                h_ = histogram_onehot(x[-1],
                                      coder.code_to_name,
                                      NOTE_NUMBER)
                h.append(norm(h_, H_TEMP))
            x = np.array(x)
            y = np.array(y)
            h = np.array(h)

            # sparse to complex
            x = to_categorical(x.flatten(), num_classes=DIM_IN).\
                reshape(x.shape[0], x.shape[1], DIM_IN)
            x[:, :, -len(NOTE_NUMBER):] = h[:, None, :]
            y = y.reshape(x.shape[0], x.shape[1], 1)
            yield (x, y)

    # model
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM_IN,
                       output_dim=DIM_OUT,
                       cells=[512, 512, 512],)
    model.build()
#   model.model.load_weights('temp/stackedrnn_hist_easy.h5')
#   model.train(data_generator(),
#               opt=1e-3,
#               steps_per_epoch=30,
#               epochs=300,
#               save_path='temp/stackedrnn_hist_easy.h5')

    model.build_generator('temp/stackedrnn_hist_easy.h5')
    res = model.generate(seed=64,
                         length=3000,
                         addition=norm(hist[1], H_TEMP),
                         verbose=0)

    events = [coder.code_to_name(note) for note in
              res[:, :DIM_IN-len(NOTE_NUMBER)].argmax(1)]
    np.set_printoptions(precision=2)
    res_hist = histogram(events, NOTE_NUMBER)
    res_hist /= res_hist.sum()
    print res_hist
    print hist[0]

    mid = Song()
    track = mid.add_track()
    # the ``useful'' notes are more than usual
    # therefore use a smaller _MIDO_TIME_SCALE
    for msgi in coder.decode(res, max_sustain=5.0):
        track.append(msgi)
    mid.save_as('simple_rnn.mid')
