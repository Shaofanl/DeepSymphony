import numpy as np

from DeepSymphony.models.StackedLSTM import (
    CondStackedLSTM, CondStackedLSTMHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.MidoCoder import ExampleCoder
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi
from DeepSymphony.utils.Player import ExampleCoderPlayer

import mido
import serial
import sys


def process(params):
    ind, l, r = params
    # [l, r)
    song = data[ind]
    sample_range = 50
    sys.stdout.flush()
    length = len(song)
    ci = np.zeros((r-l, hparam.cond_len))

    for ind in xrange(l, r):
        start = max(0, min(ind-sample_range/2, length-sample_range))
        xi = song[start: start+sample_range]

        notes = xi[xi < 128]
        if len(notes):
            # mean pitch
            #   original: 13 bits (0~12)
            #   rescaled: only take (4~8)
            mean_pitch = np.mean(notes)
            mean_pitch = np.clip(mean_pitch/10., 5., 9.)
            mean_pitch = int(np.round((mean_pitch-5.)/4.*9.))
            ci[ind-l, mean_pitch] = 1.0

            # histogram - 12 bits (0~11)
            hist = np.bincount(notes % 12,
                               minlength=12).astype('float')
            hist /= hist.sum()
            ci[ind-l, 10+10:] = hist

        # density
        #   original: 11 bits (0~10)
        #   rescaled: only take (3~5)

        # density 1:
        # density = float(len(notes)) / sample_range
        # density = np.clip(density*10., 0., 2.5)
        # density = int(np.round((density-0.)/2.5 * 9.))

        # density 2:
        rest = xi[(256 < xi)*(xi < 356)] - 256
        density = rest.sum()
        density = np.clip(density, 30., 150.)
        density = int(np.round((density-30.)/120. * 9.))
        ci[ind-l, 10+density] = 1.0
        # print mean_pitch, density, hist
    return ci


if __name__ == '__main__':
    mode = 'train'
    # mode = 'generate'
    # mode = 'jevois'

    np.random.seed(32)

    coder = ExampleCoder(return_onehot=False)
    hparam = CondStackedLSTMHParam(
        # basic
        cells=[64],  # [512, 512, 512],
        embed_dim=128,
        input_dim=coder.EVENT_COUNT,
        output_dim=coder.EVENT_COUNT,
        cond_len=32,
        # training
        batch_size=32,
        timesteps=20,
        iterations=500,
        learning_rate=1e-4,
        continued=True,
        overwrite_workdir=True,
        clip_norm=1.0,
        # generate
        gen_len=500,
        temperature=1.0,
    )
    model = CondStackedLSTM(hparam)

    global data
    try:
        data = np.load('temp/cond_easy.npz')['data']
        # data = np.load('temp/cond_e-comp.npz')['data']
    except:
        # data = np.array(map_dir(lambda fn: coder.encode(get_midi(fn)),
        #                         './datasets/easymusicnotes'))
        # np.savez('temp/cond_easy.npz', data=data)
        # data = np.array(map_dir(lambda fn: coder.encode(get_midi(fn)),
        #                         './datasets/e-comp/'))
        # np.savez('temp/cond_e-comp.npz', data=data)
        pass
    data = [data[2]]

    if mode == 'train':
        from multiprocessing import Pool
        pool = Pool(8)

        def fetch_data(batch_size):
            x, y, c = [], [], []
            indices = []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0] -
                                          hparam.timesteps-1)
                xi = data[ind][start:start+hparam.timesteps]
                yi = data[ind][start+1:start+hparam.timesteps+1]
                indices.append((ind, start+1, start+hparam.timesteps+1))
                x.append(xi)
                y.append(yi)
            c = pool.map(process, indices)
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
            cond[0] = 1.0   # 0~9
            cond[15] = 1.0  # 13~17
            cond[[20, 22, 24, 25, 27, 29, 31]] = 1./7.  # C-major
            while True:
                yield cond

        hparam.temperature = 1.0
        hparam.gen_len = 1000
        result = model.generate(
            conds=cond_generator(),
            sample=sample,
            length=hparam.gen_len,
        )
        result = coder.decode(result)
        save_midi('example.mid', result)
        save_midi('truth.mid', coder.decode(data[0]))

    elif mode == 'jevois':
        port_name = mido.get_output_names()[0]
        print 'using port', port_name
        player = ExampleCoderPlayer(outport_name=port_name,
                                    decoder=coder,
                                    max_sustain=4.,
                                    speed=1.0)
        ser = serial.Serial('/dev/ttyACM0', timeout=0.1)

        def sample(output):
            ind = np.random.choice(hparam.output_dim,
                                   p=output.flatten())

            player.play(ind)
            return ind

        from multiprocessing import Process, Array
        from time import sleep

        def cond_generator(val):
            while True:
                sleep(0.1)
                try:
                    cond = np.zeros((hparam.cond_len, ))
                    cond[[20, 22, 24, 25, 27, 29, 31]] = 1./7.  # C-major
                    ser.reset_input_buffer()
                    ser.write('fetch\n')
                    res = ser.readline().strip().split('|')
                    x, y = map(float, res)
                    mean_pitch = int(np.round((240-y)/240.*9.))
                    density = int(np.round(x/320.*9.))
                    cond[mean_pitch] = 1.0   # 0~9
                    cond[density] = 1.0  # 13~17
                    print 'pitch:', mean_pitch, 'density:', density
                except:
                    print 'miss'
                val[:] = cond.tolist()[:]
        val = Array('f', range(hparam.cond_len))
        proc = Process(target=cond_generator, args=(val,))
        proc.start()

        def cond_return():
            while True:
                yield np.array(val[:])

        result = model.generate(
            conds=cond_return(),
            sample=sample,
            length=50000,
        )
