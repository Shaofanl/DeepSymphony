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
    mode = 'myo'

    if mode == 'train':
        _timesteps = 128
    elif mode == 'generate':
        _timesteps = 640
    elif mode == 'myo':
        _timesteps = 4
    hparam = DCRNNHParam(
        # basic
        cells=[64, 32],  # [64, 32, 32],
        repeats=[1, 1],  # [8, 2, 1],
        last_bidirectional=False,
        bidirection=[False, False],
        timesteps=_timesteps,
        code_dim=50,
        vocab_size=128,
        basic_cell=rnn.LSTMCell,
        onehot=False,
        timestep_pad=False,
        code_ndim=2,
        # hparam
        trainable_gen=['generator'],
        D_lr=1e-3,
        G_lr=5e-4,  # change to 1e-4 when finetuning
        G_k=5,
        D_boost=0,
        G_clipnorm=None,  # 1.0,
        # traini
        batch_size=32,
        continued=False,
        overwrite_workdir=True,
        iterations=20000,
        workdir='./temp/DCRNN_RhythmGAN/'
    )
    model = DCRNN(hparam)
    model.build()
    # coder = NoteDurationCoder(normalize_key='C5', first_voice=False)
    coder = MultiHotCoder(normalize_key='C5', only_major=True)

    if hparam.code_dim == 3:
        def sample(batch_size):
            code_s = np.random.uniform(-1., +1.,
                                       size=(batch_size,
                                             hparam.code_dim))
            code_e = np.random.uniform(-1., +1.,
                                       size=(batch_size,
                                             hparam.code_dim))
            res = np.zeros((batch_size,
                            hparam.timesteps,
                            hparam.code_dim))
            interpolation = np.linspace(0, 1, hparam.timesteps)[None, :, None]
            res[:, :, :] = code_s[:, None, :] * interpolation + \
                code_e[:, None, :] * (1-interpolation)
            return res
    else:
        def sample(batch_size):
            return np.random.uniform(-1., +1.,
                                     size=(batch_size,
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
    # import matplotlib.pyplot as plt
    # for d in data:
    #     print d.shape
    #     for v in d:
    #         fig, ax = plt.subplots()
    #         ax.imshow(v)
    #     plt.show()
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
        seed = 619122590
        print 'seed', seed
        np.random.seed(seed)

        mode = '1'
        mode = '10'
        mode = '2x'
        if mode == '10':
            code = sample(10)
            code = np.reshape(
                np.tile(np.expand_dims(code, 1), (1, 128, 1)),
                (-1, code.shape[-1]))
            song = model.generate([code], code_img=True, img=True)[0]
        if mode == '1':
            song = model.generate(code, img=True)[0]
        if mode == '2x':
            codeA = sample(1)
            codeA = np.reshape(np.tile(np.expand_dims(codeA, 1), (1, 128, 1)),
                               (-1, codeA.shape[-1]))
            codeB = sample(1)
            codeB = np.reshape(np.tile(np.expand_dims(codeB, 1), (1, 128, 1)),
                               (-1, codeB.shape[-1]))
            code = np.concatenate([codeA, codeB, codeA, codeB, codeA], axis=0)
            song = model.generate([code], code_img=True, img=True)[0]

        print song[song.nonzero()].mean()
        final = song > -0.3
        coder.decode(final, speed=1.).write('midi', 'example.mid')

        coder = NoteDurationCoder(first_voice=False)
        note, dura = coder.encode(ms.converter.parse('example.mid'),
                                  force=True)
        coder.decode(note, 2).write('midi', 'quantized.mid')

        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.imshow(song.T[::-1, :])
        plt.subplot(212)
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

    if mode == 'myo':
        import mido
        from DeepSymphony.utils.Player import MultihotsPlayer
        from DeepSymphony.utils.Myo import Myo
        from multiprocessing import Process, Array
        import matplotlib.pyplot as plt

        port_name = mido.get_output_names()[0]
        print 'using port', port_name
        player = MultihotsPlayer(outport_name=port_name,
                                 threshold=-0.3,
                                 speed=6.)

        # rng = np.random.RandomState(10)
        rng = np.random
        noise = Array('f', rng.rand(hparam.code_dim).tolist())

        fig, ax = plt.subplots()
        plt.show(block=False)
        bars = plt.bar(np.arange(1, 9),
                       np.zeros(8),
                       color=rng.rand(8, 3),)
        ax.set_ylim([0, 2000])

        def emg_handler(emg, moving):
            print emg
            noise[0:8] = np.clip((np.array(emg)-500)/500., -1, +1)
            # noise[8:16] = np.clip((np.array(emg)-500)/500., -1, +1)
            for bar, ele in zip(bars, emg):
                bar.set_height(ele)
            fig.canvas.draw()

        myo = Myo()
        myo.add_emg_handler(emg_handler)
        myo.connect()

        def myo_process():
            try:
                while True:
                    myo.run(1)
            except KeyboardInterrupt:
                pass
            finally:
                print 'exiting myo...'
                myo.disconnect()
        proc = Process(target=myo_process)
        proc.start()

        def code_generator():
            while True:
                n = np.array(noise[:])
                n = np.tile(np.expand_dims(np.expand_dims(n, 0), 0), (1, 4, 1))
                yield n

        def callback(seqs):
            for seq in seqs[0]:
                player.play(seq)

        song = model.generate(code_generator(),
                              real_time_callback=callback,
                              code_img=True,
                              img=True)
