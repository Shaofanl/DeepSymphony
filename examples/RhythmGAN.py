# experimental
import numpy as np

from DeepSymphony.models.SeqGAN import (
    SeqGAN, SeqGANHParam
)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
import music21 as ms
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn


if __name__ == '__main__':
    mode = 'train'

    hparam = SeqGANHParam(
        # basic
        cells=[64, 32],
        timesteps=128,
        code_dim=200,
        vocab_size=128,
        basic_cell=rnn.GRUCell,
        # hparam
        D_lr=1e-4,
        D_boost=0,
        G_lr=1e-3,
        G_k=5,
        G_clipnorm=1.0,
        # train
        batch_size=25,
        continued=False,
        overwrite_workdir=True,
        iterations=50000,
        workdir='./temp/RhythmGAN/'
    )
    model = SeqGAN(hparam)
    model.build()
    coder = NoteDurationCoder(normalize_key='C5', first_voice=True)

    try:
        # raise Exception
        data = np.load('temp/easy.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn))[0],
            './datasets/easymusicnotes/'))
        np.savez('temp/easy.npz', data=data)

    print(len(data), map(lambda x: len(x), data))
    data = filter(lambda x: len(x) > hparam.timesteps, data)
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

    def sample(batch_size):
        return np.random.normal(loc=0.,
                                scale=1.,
                                size=(hparam.batch_size,
                                      hparam.code_dim))

    fetch_train_data = fetch_data_g(train_data)
    fetch_test_data = fetch_data_g(test_data)

    if mode == 'train':
        model.train(sample,
                    fetch_train_data,
                    continued=hparam.continued)
