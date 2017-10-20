import numpy as np
import music21 as ms

from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import MultiHotCoder
from DeepSymphony.models.MultivoiceLSTM import (
    MultivoiceLSTM, MultivoiceLSTMHParam
)


if __name__ == '__main__':
    mode = 'train'

    hparam = MultivoiceLSTMHParam(batch_size=64,
                                  nb_voices=2,
                                  cells=[256, 256],
                                  timesteps=100,
                                  iterations=3000,
                                  overwrite_workdir=True)
    model = MultivoiceLSTM(hparam)

    coder = MultiHotCoder(bits=128)
    data = np.array(map_dir(lambda fn: coder.encode(ms.converter.parse(fn)),
                            './datasets/easymusicnotes/level6/'))
    data = filter(lambda x: x.shape[0] == 2, data)

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
    model.train(fetch_data)
