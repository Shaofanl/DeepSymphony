import numpy as np
import music21 as ms

from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import MultiHotCoder
from DeepSymphony.models.MultivoiceLSTM import (
    MultivoiceLSTM, MultivoiceLSTMHParam
)


if __name__ == '__main__':
    mode = 'train'
    # mode = 'generate'

    if mode == 'train':
        hparam = MultivoiceLSTMHParam(batch_size=64,
                                      b_cells=[128],
                                      m_cells=[256],
                                      t_cells=[128],
                                      nb_voices=2,
                                      timesteps=100,
                                      iterations=1000,
                                      learning_rate=1e-3,
                                      overwrite_workdir=True)
        model = MultivoiceLSTM(hparam)

        coder = MultiHotCoder(bits=128)
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn)),
            './datasets/easymusicnotes/')
        )
        data = filter(lambda x: x.shape[0] >= 2, data)
        data = map(lambda x: x[:2], data)

        def fetch_data(batch_size):
            x = [[] for _ in range(hparam.nb_voices)]
            y = [[] for _ in range(hparam.nb_voices)]
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[1] -
                                          hparam.timesteps-1)
                for vind in range(hparam.nb_voices):
                    x[vind].append(
                        data[ind][vind][start:start+hparam.timesteps]
                    )
                    y[vind].append(
                        data[ind][vind][start+1:start+hparam.timesteps+1]
                    )
            x, y = np.array(x), np.array(y)
            return x, y
        model.train(fetch_data, continued=True)
    elif mode == 'generate':
        hparam = MultivoiceLSTMHParam(batch_size=1,
                                      b_cells=[128],
                                      m_cells=[256],
                                      t_cells=[128],
                                      nb_voices=2,
                                      timesteps=1)
        THRESHOLD = 0.8
        coder = MultiHotCoder()
        model = MultivoiceLSTM(hparam)
        maxp = []
        boosting_count = 5

        def handle(output):
            # noisy = np.random.uniform(0.8, 1/0.8, size=output.shape)
            # output *= noisy
            global boosting_count, maxp
            maxp.append(output.max())
            if boosting_count > 0:  # easy on the first ones
                output[output < 0.3] = 0
                boosting_count -= 1
            else:
                output[output < THRESHOLD] = 0
            output[output != 0] = 1.0
            return output

        GEN_LEN = 100
        result = model.generate(length=GEN_LEN,
                                handle=handle)
        print maxp
        result = result.reshape(hparam.nb_voices,
                                GEN_LEN,
                                hparam.output_dim)

        import matplotlib.pyplot as plt
        plt.imshow(result.sum(0).T)
        plt.colorbar()
        plt.savefig('example.png')

        stream = coder.decode(result)
        stream.write('midi', 'example.mid')
