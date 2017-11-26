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
    # mode = 'myo'

    if mode in ['train', 'collect']:
        _timesteps = 64
    elif mode == 'generate':
        _timesteps = 640
    elif mode == 'myo':
        _timesteps = 16
    hparam = DCRNNHParam(
        # basic
        cells=[64, 32],  # [64, 32, 32],
        repeats=[1, 1],  # [8, 2, 1],
        bidirection=[False, False],
        timesteps=_timesteps,
        code_dim=50,
        vocab_size=128,
        basic_cell=rnn.LSTMCell,
        onehot=False,
        timestep_pad=False,
        code_ndim=2,
        # deconv_decision=True,
        conditional=True,
        cond_dim=12,
        # hparam
        trainable_gen=['generator'],
        D_lr=1e-3,
        G_lr=8e-4,  # change to 1e-4 when finetuning
        G_k=2,
        D_boost=0,
        G_clipnorm=None,  # 1.0,
        # traini
        batch_size=32,
        continued=False,
        overwrite_workdir=True,
        iterations=40000,
        workdir='./temp/DCRNN_cond/'
    )
    model = DCRNN(hparam)
    model.build()
    # coder = NoteDurationCoder(normalize_key='C5', first_voice=False)
    # coder = MultiHotCoder(normalize_key='C5', only_major=True)
    coder = MultiHotCoder(# normalize_key='C5',
                          with_velocity=True,
                          # only_major=True,
                          length_limit=np.inf)

    try:
        # data = np.load('temp/easy.npz')
        # raise Exception
        data = np.load('temp/piano-midi.npz')
        voi, vel = data['voice'], data['velocity']
        # data = (voi*vel)/127.
        data = voi
    except:
        def read_song(filename):
            return coder.encode(ms.converter.parse(filename))

        data = np.array(map_dir(
            read_song,
            './datasets/piano-midi.de/', cores=8))
            # './datasets/easymusicnotes/', cores=8))
        data = filter(lambda x: x is not None, data)
        voi, vel = zip(*data)
        np.savez('temp/easy.npz', voice=voi, velocity=vel)
        data = voi
        # np.savez('temp/piano-midi.npz', voice=voi, velocity=vel)

    print(len(data), map(lambda x: len(x), data))
    data = filter(lambda x: len(x) > 0 and x.shape[0] > hparam.timesteps, data)
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
                # seq[:, :60] = 0
                seqs.append(seq)
            seqs = np.array(seqs)
            seqs = seqs*2-1
            return seqs
        return fetch_data

    fetch_train_data = fetch_data_g(train_data)
    fetch_test_data = fetch_data_g(test_data)

    def cond_vfun(arr):
        cond = np.zeros((arr.shape[0], hparam.cond_dim))
        arr = ((arr+1)/2.).mean(1)
        for i in range(12):
            cond[:, i] = arr[:, i::12].sum(1)
        sums = cond.sum(1)
        sums[sums == 0] = 1
        cond = cond/sums[:, None]
        return cond

    seqs = fetch_train_data(5000)
    cond_collections = cond_vfun(seqs)
    print cond_collections
    if hparam.code_dim == 3:
        raise NotImplemented
    else:
        def sample(batch_size):
            code = np.random.uniform(-1., +1.,
                                     size=(batch_size,
                                           hparam.code_dim))
            # cond = np.random.rand(batch_size, hparam.cond_dim)
            # cond = cond/cond.sum(1)[:, None]
            cond = cond_collections[np.random.choice(
                5000, size=(batch_size,), replace=False)]
            return code, cond

    if mode == 'train':
        pass
        model.train(sample, fetch_train_data, cond_vfun=cond_vfun,
                    continued=hparam.continued)
    if mode == 'generate':
        seed = np.random.randint(1e+9)
        # seed = 619122590
        print 'seed', seed
        np.random.seed(seed)

        mode = '1'
        # mode = 'rand'
        mode = '2x'
        cond = [cond_collections[0]]
        # cond = np.array([[0, 0.2, 0, 0.2, 0,
        # 0, 0.2, 0, 0.2, 0, 0.2, 0]])
        print cond
        if mode == '1':
            code, _ = sample(1)
            print code
            song = model.generate(code, cond, img=True)[0]
        if mode == 'rand':
            codes = []
            code, _ = sample(1)
            for i in range(20):
                code = np.clip(np.random.uniform(-1, +1, size=code.shape),
                               -1, +1)
                codes.append(np.reshape(
                    np.tile(
                        np.expand_dims(
                            np.concatenate([code, cond], axis=-1),
                            1),
                        (1, 32, 1)),
                    (1, 32, hparam.code_dim+hparam.cond_dim)))
                print map(lambda x: x.shape, codes)
            codes = np.concatenate(codes, 1)
            song = model.generate(codes, cond=cond, code_img=True, img=True)[0]

        if mode == '2x':
            codeA = np.concatenate(sample(1), -1)
            codeA = np.reshape(np.tile(np.expand_dims(codeA, 1), (1, 128, 1)),
                               (-1, codeA.shape[-1]))
            codeB = np.concatenate(sample(1), -1)
            codeB = np.reshape(np.tile(np.expand_dims(codeB, 1), (1, 128, 1)),
                               (-1, codeB.shape[-1]))
            codeC = np.concatenate(sample(1), -1)
            codeC = np.reshape(np.tile(np.expand_dims(codeC, 1), (1, 64, 1)),
                               (-1, codeC.shape[-1]))

            code = np.concatenate([codeA, codeB, codeA, codeB, codeA],
                                  axis=0)
            song = model.generate([code], cond=cond,
                                  code_img=True, img=True)[0]

        print song[song.nonzero()].mean()
        final = song > 0.9
        velocity = ((song+1)/2.*127).astype('uint8')
        print 'velocity range', velocity.max(), velocity.min()
        coder.decode(final, speed=1.).\
            write('midi', 'example.mid')

        coder = NoteDurationCoder(first_voice=False)
        note, dura = coder.encode(ms.converter.parse('example.mid'),
                                  force=True)
        coder.decode(note, duracode=2).write('midi', 'quantized.mid')

        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.imshow(song.T[::-1, :])
        plt.subplot(212)
        plt.imshow(final.T[::-1, :])
        # plt.colorbar(orientation='horizontal')
        plt.savefig('example.png')
        plt.show()
