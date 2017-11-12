# experimental
import numpy as np

from DeepSymphony.models.SeqGAN import (
    DCRNN, DCRNNHParam
)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import (
    NoteDurationCoder, MultiHotCoder
)
import music21 as ms
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn


if __name__ == '__main__':
    mode = 'train'
    mode = 'generate'
    # mode = 'analyze'

    hparam = DCRNNHParam(
        # basic
        cells=[64, 32],
        repeats=[8, 2],
        timesteps=512,
        code_dim=100,
        vocab_size=128,
        basic_cell=rnn.LSTMCell,
        onehot=False,
        # hparam
        trainable_gen=['generator'],
        D_lr=1e-3,
        G_lr=1e-4,
        G_k=5,
        D_boost=0,
        G_clipnorm=1.0,
        # train
        batch_size=32,
        continued=True,
        overwrite_workdir=True,
        iterations=20000,
        workdir='./temp/DCRNN_RhythmGAN/'
    )
    model = DCRNN(hparam)
    model.build()
    # coder = NoteDurationCoder(normalize_key='C5', first_voice=False)
    coder = MultiHotCoder(normalize_key='C5')

    def sample(batch_size):
        # return np.random.normal(loc=0.,
        #                         scale=1.,
        return np.random.uniform(-1., +1.,
                                 size=(hparam.batch_size,
                                       hparam.code_dim))

    try:
        data = np.load('temp/easy.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn)),
            './datasets/easymusicnotes/'))
        np.savez('temp/easy.npz', data=data)

    print(len(data), map(lambda x: len(x), data))
    data = filter(lambda x: len(x) > 0 and x.shape[1] > hparam.timesteps, data)
    data = map(lambda x: x.sum(0), data)
    print(len(data), map(lambda x: len(x), data))

    train_data, test_data = train_test_split(data,
                                             test_size=0.22,
                                             random_state=32)

    def fetch_data_g(dataset):
        def fetch_data(batch_size):
            seqs = []
            for _ in range(batch_size):
                ind = np.random.randint(len(dataset))
                start = np.random.randint(dataset[ind].shape[0] -
                                          hparam.timesteps-1)
                seq = dataset[ind][start:start+hparam.timesteps]
                seqs.append(seq)
            return np.array(seqs)
        return fetch_data

    fetch_train_data = fetch_data_g(train_data)
    fetch_test_data = fetch_data_g(test_data)

    if mode == 'train':
        model.train(sample,
                    fetch_train_data,
                    continued=hparam.continued)
    if mode == 'analyze':
        model.analyze(sample)

    if mode == 'generate':
        song = model.generate(sample(1), img=True)[0]
        song = song>0.5
        coder.decode(song, speed=1.).write('midi', 'example.mid')

        import matplotlib.pyplot as plt
        plt.imshow(song)
        plt.colorbar()
        plt.show()
        plt.savefig('example.png')

        # t, p = (song > 0.80).nonzero()
        # song = []
        # last_t = 0
        # duration = 2
        # for ti, pi in zip(t, p):
        #     if last_t < ti:
        #         song.append(128)
        #     song.append(pi)
        #     last_t = ti
        # print song
        # coder.decode(song, 2).write('midi', 'example.mid')
