import tensorflow as tf
import numpy as np

from DeepSymphony.models.SeqAE import (
    ContinuousSeqAE, ContinuousSeqAEHParam)
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
from DeepSymphony.utils.MidoCoder import ExampleCoder
from DeepSymphony.utils.sample import continuous_sample
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi
from DeepSymphony.utils.SparseCodeCompressor import SparseCodeCompressor


if __name__ == '__main__':
    # coder = NoteDurationCoder(normalize_key='C5',
    #                           # single=True,
    #                           first_voice=True)
    coder = ExampleCoder(shift_count=10)
    # song = 'datasets/easymusicnotes/level12/tonight-i-wanna-cry-keith-urban-country-piano-level-12.mid'
    song = 'Nuvole-Bianche.mid'

    song = coder.encode(get_midi(song))
    # song = song[song != coder.event_to_code(0, 3)]
    compressor = SparseCodeCompressor(song)
    song = compressor.to_compact(song)
    # save_midi('truth.mid', coder.decode(compressor.to_sparse(song)))

    mode = 'train'
    mode = 'rec'
    mode = 'random_walk'  # can use different timestep when generating

    hparam = ContinuousSeqAEHParam(batch_size=32,
                                   encoder_cells=[512, 32],
                                   decoder_cells=[32, 512],
                                   timesteps=16,
                                   gen_timesteps=16,
                                   embed_dim=512,
                                   basic_cell=tf.contrib.rnn.GRUCell,
                                   learning_rate=1e-3,
                                   iterations=1000,
                                   continued=True,
                                   only_train_quantized_rec=True,
                                   vocab_size=compressor.count+1,
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
        seqs = []
        for _ in range(batch_size):
            start = np.random.randint(song.shape[0] -
                                      hparam.timesteps-1)
            seq = song[start:start+hparam.timesteps]
            seqs.append(seq)
        return np.array(seqs)

    def fetch_tri_data(batch_size):
        src_seqs, nei_seqs, int_seqs = [], [], []
        for _ in range(batch_size):
            start = np.random.randint(song.shape[0] -
                                      hparam.timesteps-1-3*hparam.timesteps)
            src_seq = song[start+0*hparam.timesteps:start+1*hparam.timesteps]
            nei_seq = song[start+1*hparam.timesteps:start+2*hparam.timesteps]
            int_seq = song[start+2*hparam.timesteps:start+3*hparam.timesteps]
            src_seqs.append(src_seq)
            nei_seqs.append(nei_seq)
            int_seqs.append(int_seq)
        return (np.array(src_seqs), np.array(nei_seqs), np.array(int_seqs))

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

        song = compressor.to_sparse(song)
        rec = compressor.to_sparse(rec)
        print rec[:100]
        save_midi('truth.mid', coder.decode(song, _MIDO_TIME_SCALE=1.5))
        save_midi('rec.mid', coder.decode(rec, _MIDO_TIME_SCALE=1.5))
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
        gen = compressor.to_sparse(gen)
        save_midi('example.mid', coder.decode(gen, _MIDO_TIME_SCALE=1.2))
