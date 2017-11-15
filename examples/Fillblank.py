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
        cells=[64, 32],  # [64, 32, 32],
        repeats=[1, 1],  # [8, 2, 1],
        last_bidirectional=False,
        timesteps=256 if mode == 'train' else 1024,
        code_dim=128,
        vocab_size=128,
        basic_cell=rnn.LSTMCell,
        onehot=False,
        code_ndim=3,
        timestep_pad=False,
        plus_code=False,
        show_grad=False,
        show_input=True,
        # hparam
        trainable_gen=['generator'],
        D_lr=1e-3,
        G_lr=2e-4,  # change to 1e-4 when finetuning
        G_k=5,
        D_boost=0,
        G_clipnorm=None,  # 1.0,
        # traini
        batch_size=32,
        continued=True,
        overwrite_workdir=True,
        iterations=20,
        workdir='./temp/Fillblank/'
    )
    model = DCRNN(hparam)
    model.build()
    # coder = NoteDurationCoder(normalize_key='C5', first_voice=False)
    coder = MultiHotCoder(normalize_key='C5')

    try:
        data = np.load('temp/easy.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn)),
            './datasets/easymusicnotes/'))
        np.savez('temp/easy.npz', data=data)

    print(len(data), map(lambda x: len(x), data))
    data = filter(lambda x: len(x) > 0 and x.shape[1] > hparam.timesteps, data)
    # import matplotlib.pyplot as plt
    # for d in data:
    #     print d.shape
    #     for v in d:
    #         fig, ax = plt.subplots()
    #         ax.imshow(v)
    #     plt.show()
    data = map(lambda x: x.sum(0), data)
    print(len(data), map(lambda x: len(x), data))

    def sample(batch_size, noise=1.00):
        seqs = []
        for _ in range(batch_size):
            ind = np.random.randint(len(data))
            start = np.random.randint(data[ind].shape[0] -
                                      hparam.timesteps-1)
            seq = data[ind][start:start+hparam.timesteps]
            # seq[:, :60] = 0
            seqs.append(seq)
        seqs = np.array(seqs)
        seqs = seqs*np.random.binomial(1, 1-noise, size=seqs.shape)
        return seqs

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
                # seq[:, :60] = 0
                seqs.append(seq)
            seqs = np.array(seqs)
            seqs = seqs*2-1
            return seqs
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
        seed = np.random.randint(1e+9)
        # seed = 292251089
        print 'seed', seed
        np.random.seed(seed)
        code = sample(1, noise=0.50)
        song = model.generate(code, img=True)[0]
        print song[song.nonzero()].mean()
        final = song > -0.8
        coder.decode(final, speed=1.).write('midi', 'example.mid')

        print final.nonzero()

        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.imshow(code[0].T[::-1, :])
        plt.subplot(312)
        plt.imshow(song.T[::-1, :])
        plt.subplot(313)
        plt.imshow(final.T[::-1, :])
        # plt.colorbar(orientation='horizontal')
        plt.savefig('example.png')
        plt.show()

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
