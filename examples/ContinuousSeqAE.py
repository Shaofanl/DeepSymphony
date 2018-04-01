# experimental
import numpy as np
import tensorflow as tf

from DeepSymphony.models.SeqAE import (
    ContinuousSeqAE, ContinuousSeqAEHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
import music21 as ms
from sklearn.model_selection import train_test_split
from DeepSymphony.utils.sample import continuous_sample
from DeepSymphony.utils.sample import remove_con_dup


if __name__ == '__main__':
    # usage:
    # 1. train a model
    # 2. evaluate it if you want
    # 3. collect the codes
    # 4. generate with the collected code
    mode = 'train'
    # mode = 'eval'
    # mode = 'continuous_eval'
    # mode = 'rec'
    # mode = 'plot'
    mode = 'random_walk'
    # mode = 'shift'

    hparam = ContinuousSeqAEHParam(batch_size=128,
                                   encoder_cells=[256, 16],
                                   decoder_cells=[16, 256],
                                   timesteps=8,
                                   gen_timesteps=8,
                                   embed_dim=512,
                                   basic_cell=tf.contrib.rnn.GRUCell,
                                   learning_rate=1e-3,
                                   iterations=2000,
                                   continued=False,
                                   only_train_quantized_rec=False,
                                   vocab_size=128+1,
                                   debug=False,
                                   overwrite_workdir=True,
                                   # workdir='./temp/SeqGAN_ablation/',
                                   clip_norm=1.,
                                   alpha=5e-3,  # 5-->2, alpha*con_loss
                                   beta=1.00,
                                   gamma=1e-3)  # gamma*q_loss
    model = ContinuousSeqAE(hparam)
    model.build()
    # coder = ExampleCoder()
    coder = NoteDurationCoder(normalize_key='C5',)
                              # single=True,
                              # first_voice=True)

    try:
        data = np.load('temp/easy.npz')['data']
        # data = np.load('temp/1138.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn))[0],
            './datasets/1138ChinesePopSongs/1138/'))
        np.savez('temp/1138.npz', data=data)

    print(len(data), map(lambda x: len(x), data))
    data = filter(lambda x: len(x) > hparam.timesteps, data)
    data = map(remove_con_dup, data)
    print(len(data), map(lambda x: len(x), data))

    train_data, test_data = train_test_split(data,
                                             test_size=0.22,
                                             random_state=32)

    def fetch_tri_data_g(dataset):
        def fetch_tri_data(batch_size):
            src_seqs, nei_seqs, int_seqs = [], [], []
            for _ in range(batch_size):
                ind = np.random.randint(len(dataset))
                start = np.random.randint(dataset[ind].shape[0] -
                                          hparam.timesteps-1-3*hparam.timesteps)
                src_seq = dataset[ind][start+0*hparam.timesteps:start+1*hparam.timesteps]
                nei_seq = dataset[ind][start+1*hparam.timesteps:start+2*hparam.timesteps]
                int_seq = dataset[ind][start+2*hparam.timesteps:start+3*hparam.timesteps]
                src_seqs.append(src_seq)
                nei_seqs.append(nei_seq)
                int_seqs.append(int_seq)
            return (np.array(src_seqs),
                    np.array(nei_seqs),
                    np.array(int_seqs))
        return fetch_tri_data

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
    fetch_train_tri_data = fetch_tri_data_g(train_data)
    fetch_test_tri_data = fetch_tri_data_g(test_data)

    fetch_one_data = fetch_data_g([train_data[1]])
    fetch_one_tri_data = fetch_tri_data_g([train_data[1]])

    if mode == 'train':
        model.train(fetch_one_data,
                    fetch_one_tri_data,
                    continued=hparam.continued)
    if mode == 'collect':
        np.random.seed(32)
        collection, seqs = model.collect(fetch_train_data, samples=10)
        np.savez(hparam.workdir+'code_collection.npz',
                 wrapper={'code': collection, 'seqs': seqs})
    if mode == 'eval':
        print('trainset')
        seqs = fetch_train_data(hparam.batch_size)
        pred, train_pred = model.eval(seqs)
        np.set_printoptions(linewidth=np.inf)
        for i in range(5):
            print('='*200)
            print(seqs[i])
            print(train_pred[i])
            print(pred[i][:2*hparam.timesteps])

        print('testset')
        seqs = fetch_test_data(hparam.batch_size)
        pred, train_pred = model.eval(seqs)
        np.set_printoptions(linewidth=np.inf)
        for i in range(5):
            print('='*200)
            print(seqs[i])
            print(train_pred[i])
            print(pred[i][:2*hparam.timesteps])
    if mode == 'continuous_eval':
        np.set_printoptions(precision=3)
        seq = train_data[1].copy()
        batch = continuous_sample(seq,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        code = model.encode(batch[:hparam.batch_size], quantized=True)
        print((code[:-1] != code[1:]).sum(1))
        print(code[0])
        print(code[1])
        print((code[0][None, :] != code).sum(1))
        print((code[1][None, :] != code).sum(1))
        print((code[2][None, :] != code).sum(1))
        print((code[3][None, :] != code).sum(1))
        print(code.min())
        print(code.max())
        print(code.mean())

        # for c in code:
            # print np.histogram(c)

    if mode == 'rec':
        seq = train_data[1].copy()
        batch = continuous_sample(seq,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        ori = code = model.encode(batch, quantized=True)
        print(code)
        rec = model.generate(code, quantized=True).flatten()
        print(seq.shape)
        print(batch.shape)
        print(rec.shape)

        coder.decode(seq, 2).write('midi', 'truth.mid')
        coder.decode(rec, 2).write('midi', 'rec.mid')

    if mode == 'plot':
        seq = train_data[0].copy()
        batch = continuous_sample(seq,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        code = model.encode(batch, quantized=True)  # first layer
        print(code)

        from sklearn.manifold import TSNE
        tsne = TSNE(2)
        code = tsne.fit_transform(code)
        print(code.shape)
        import matplotlib.pyplot as plt
        plt.plot(code[:, 0], code[:, 1],
                 marker='o', ms=8)
        for ind, (x, y) in enumerate(code):
            plt.text(x, y, str(ind))
        plt.show()

    if mode == 'random_walk':
        np.set_printoptions(precision=3)
        seq = train_data[1].copy()
        batch = continuous_sample(seq,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        code = model.encode(batch, quantized=True)
        # code = model.encode([[60, 128, 60, 128, 64, 128, 64, 128]],
        #                     quantized=True)

        rng = np.random.RandomState(30)
        pos = code[0]
        _theme = code[0].copy()
        # _theme2 = code[18].copy()
        theme = _theme
        gen = [pos.copy()]
        # pos = np.random.binomial(2, 0.5, size=(len(code[0],)))/2.
        for i in range(500):
            # rand = np.random.normal(size=(len(pos),))
            # rand /= np.sqrt((rand**2).sum())
            # pos = np.clip(pos + rand, 0, 1)

            #if rng.rand() < 0.5:
            #     pos = -pos
            # else:
            randint = rng.choice(len(theme), size=(2,), replace=False)
            pos[randint] = -pos[randint]

            # if np.random.rand() < 0.05:
            #    # shift of theme
            #    theme = pos.copy()

            # if i % 4 == 0:
            #     if np.random.rand() < 0.10:
            #         _theme = pos.copy()
            #     _theme, _theme2 = _theme2, _theme
            #     pos = _theme.copy()

            if i % 4 == 0:
                if np.random.rand() < 0.80:
                    pos = theme.copy()
                if np.random.rand() < 0.20:
                    print('switch theme')
                    theme = pos.copy()

            # if np.random.rand() < 0.50:
            #     randint = np.random.randint(len(theme), size=(8,))
            #     pos[randint] = -pos[randint]

            # if rng.rand() < 0.10:
            #     print 'repeat'
            #     gen.extend(gen[-4:])

            gen.append(pos.copy())
            # print pos
        gen = np.array(gen)

        song = model.generate(gen, quantized=True).flatten()
        np.savez('temp/oneshot_example.npz', song=song)
        coder.decode(song, 2).write('midi', 'example.mid')

    if mode == 'shift':
        seq = train_data[0].copy()
        batch = continuous_sample(seq,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        code = model.encode(batch, quantized=True)[0]
        shift = np.random.choice(len(code[0]))
        code[:, shift] = -1
        print(code)
        rec = model.generate(code, quantized=True).flatten()

        coder.decode(seq, 2).write('midi', 'truth.mid')
        coder.decode(rec, 2).write('midi', 'shift.mid')

    try:
        from gi.repository import Notify
        Notify.init("ContinuousSeqAE")
        notification = Notify.Notification.new("Done")
        notification.show()
    except:
        pass
