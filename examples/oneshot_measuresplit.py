import tensorflow as tf
import numpy as np

from DeepSymphony.models.SeqAE import (
    ContinuousSeqAE, ContinuousSeqAEHParam)
from DeepSymphony.utils.Music21Coder import MeasureSplitCoder
from DeepSymphony.utils.sample import continuous_sample
import music21 as ms
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi


if __name__ == '__main__':
    coder = MeasureSplitCoder()

    song = 'datasets/easymusicnotes/level12/tonight-i-wanna-cry-keith-urban-country-piano-level-12.mid'
    # song = 'Nuvole-Bianche.mid'
    try:
        data = np.load('temp/oneshot_measuresplit.npz')
        song_note, song_dura = data['song_note'], data['song_dura']
    except:
        print 'parsing'
        song_note, song_dura = coder.encode(ms.converter.parse(song))
        np.savez('temp/oneshot_measuresplit.npz',
                 song_note=song_note, song_dura=song_dura)
    import ipdb
    ipdb.set_trace()

    mode = 'train'
    # mode = 'rec'
    # mode = 'random_walk'  # can use different timestep when generating

    hparam = ContinuousSeqAEHParam(batch_size=32,
                                   encoder_cells=[512, 32],
                                   decoder_cells=[32, 512],
                                   timesteps=None,
                                   gen_timesteps=None,
                                   embed_dim=512,
                                   basic_cell=tf.contrib.rnn.GRUCell,
                                   learning_rate=1e-3,
                                   iterations=1000,
                                   continued=True,
                                   only_train_quantized_rec=True,
                                   vocab_size=128+1,
                                   debug=False,
                                   overwrite_workdir=True,
                                   workdir='temp/Oneshot/',
                                   clip_norm=1.,
                                   alpha=1e-4,  # 5-->2, alpha*con_loss
                                   beta=1.00,
                                   gamma=2e-2)  # gamma*q_loss
    model = ContinuousSeqAE(hparam)
    model.build()

    def fetch_data(batch_size):
        inds = np.random.choice(
            len(song_note), size=(batch_size,), replace=False)
        return song_note[inds]

    def fetch_tri_data(batch_size):
        inds = np.random.choice(
            len(song_note)-2, size=(batch_size,), replace=False)
        return song_note[inds], song_note[inds+1], song_note[inds+2]

    if mode == 'train':
        model.train(fetch_data,
                    fetch_tri_data,
                    continued=hparam.continued)
    if mode == 'rec':
        batch = continuous_sample(song,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        ori = code = model.encode(batch, quantized=True)
        print code
        rec = model.generate(code, quantized=True).flatten()
        print rec[:100]
        coder.decode(song).save('midi', 'truth.mid')
        coder.decode(rec).save('midi', 'rec.mid')
    if mode == 'random_walk':
        np.set_printoptions(precision=3)
        batch = continuous_sample(song,
                                  hparam.timesteps,
                                  stride=hparam.timesteps)
        code = model.encode(batch, quantized=True)
        dis = (np.abs(code[:-1] - code[1:]).sum(1)/2).astype('int')
        print dis

        rng = np.random.RandomState(32)
        pos = code[1].copy()
        gen = [pos.copy()]
        theme = pos.copy()
        for i in range(len(dis)):
            randint = rng.choice(len(pos),
                                 size=(dis[i],),
                                 replace=False)
            pos[randint] = -pos[randint]

            if i % 4 == 0 and rng.rand() < 0.5:
                pos = theme.copy()
                if rng.rand() < 0.1:
                    theme = pos.copy()

            if rng.rand() < 0.10:
                gen.extend(gen[-4:])

            gen.append(pos.copy())
            print pos
        gen = np.array(gen)
        gen = model.generate(gen, quantized=True).flatten()
        coder.decode(gen).save('midi', 'example.mid')
