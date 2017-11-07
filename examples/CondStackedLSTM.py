import numpy as np

from DeepSymphony.models.StackedLSTM import (
    CondStackedLSTM, CondStackedLSTMHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.MidoCoder import ExampleCoder
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi
from DeepSymphony.utils.Player import ExampleCoderPlayer

import mido
import serial


if __name__ == '__main__':
    # mode = 'train'
    mode = 'generate'
    mode = 'jevois'

    np.random.seed(32)

    coder = ExampleCoder(return_onehot=False)
    hparam = CondStackedLSTMHParam(
        # basic
        cells=[512, 512, 512],
        embed_dim=300,
        input_dim=coder.EVENT_COUNT,
        output_dim=coder.EVENT_COUNT,
        cond_len=32,
        # training
        batch_size=64,
        timesteps=30,
        iterations=1000,
        learning_rate=5e-4,
        continued=True,
        overwrite_workdir=True,
        clip_norm=3.0,
        # generate
        gen_len=500,
        temperature=1.0,
    )
    model = CondStackedLSTM(hparam)

    try:
        data = np.load('temp/cond_easy.npz')['data']
    except:
        data = np.array(map_dir(lambda fn: coder.encode(get_midi(fn)),
                                './datasets/easymusicnotes'))
        np.savez('temp/cond_easy.npz', data=data)

    if mode == 'train':
        def fetch_data(batch_size):
            x, y, c = [], [], []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0] -
                                          hparam.timesteps-1)
                xi = data[ind][start:start+hparam.timesteps]
                yi = data[ind][start+1:start+hparam.timesteps+1]
                x.append(xi)
                y.append(yi)

                # calc conditional
                ci = np.zeros((1, hparam.cond_len))
                notes = xi[xi < 128]
                # mean pitch
                #   original: 13 bits (0~12)
                #   rescaled: only take (4~8)
                mean_pitch = np.mean(notes)
                mean_pitch = np.clip(mean_pitch/10., 4., 8.)
                mean_pitch = int(np.round((mean_pitch-4.)/4.*9.))
                # density
                #   original: 11 bits (0~10)
                #   rescaled: only take (3~5)
                density = float(len(notes)) / hparam.timesteps
                density = np.clip(density*10., 2., 5.)
                density = int(np.round((density-2.)/3. * 9.))
                # histogram - 12 bits (0~11)
                hist = np.bincount(notes % 12, minlength=12).astype('float')
                hist /= hist.sum()
                # fill the blank
                ci[0, mean_pitch] = 1.0
                ci[0, 10+density] = 1.0
                ci[0, 10+10:] = hist
                # print mean_pitch, density, hist
                ci = np.tile(ci, (hparam.timesteps, 1))

                c.append(ci)
            x, y, c = np.array(x), np.array(y), np.array(c)
            return x, y, c

        model.train(fetch_data)

    elif mode == 'generate':
        def sample(output):
            ind = np.random.choice(hparam.output_dim,
                                   p=output.flatten())
            return ind

        def cond_generator():
            cond = np.zeros((hparam.cond_len, ))
            cond[5] = 1.0   # 0~9
            cond[17] = 1.0  # 13~17
            cond[[20, 22, 24, 25, 27, 29, 31]] = 1.0  # C-major
            while True:
                yield cond

        result = model.generate(
            conds=cond_generator(),
            sample=sample,
            length=hparam.gen_len,
        )
        result = coder.decode(result)
        save_midi('example.mid', result)

    elif mode == 'jevois':
        port_name = mido.get_output_names()[0]
        print 'using port', port_name
        player = ExampleCoderPlayer(outport_name=port_name,
                                    decoder=coder,
                                    max_sustain=4.,
                                    speed=1.5)
        ser = serial.Serial('/dev/ttyACM0', timeout=0.01)

        def sample(output):
            ind = np.random.choice(hparam.output_dim,
                                   p=output.flatten())

            player.play(ind)
            return ind

        def cond_generator():
            last_cond = np.zeros((hparam.cond_len, ))
            while True:
                cond = np.zeros((hparam.cond_len, ))
                cond[[20, 22, 24, 25, 27, 29, 31]] = 1./7.  # C-major
                try:
                    ser.reset_input_buffer()
                    ser.write('fetch\n')
                    res = ser.readline().strip().split('|')
                    x, y = map(float, res)
                    mean_pitch = int(np.round((240-y)/240.*9.))
                    density = int(np.round(x/320.*9.))
                    cond[mean_pitch] = 1.0   # 0~9
                    cond[density] = 1.0  # 13~17
                    print 'pitch:', mean_pitch, 'density:', density
                    last_cond = cond.copy()
                except:
                    print 'miss'
                    cond = last_cond
                yield cond

        result = model.generate(
            conds=cond_generator(),
            sample=sample,
            length=50000,
        )
