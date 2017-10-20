import numpy as np

from DeepSymphony.models.StackedLSTM import (
    StackedLSTM, StackedLSTMHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.MidoCoder import ExampleCoder
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi


if __name__ == '__main__':
    # mode = 'train'
    mode = 'generate'
    if mode == 'train':
        hparam = StackedLSTMHParam(batch_size=64,
                                   cells=[256, 256],
                                   input_dim=363,
                                   timesteps=100,
                                   output_dim=363,
                                   learning_rate=5e-3,
                                   iterations=3000,
                                   overwrite_workdir=True)
        model = StackedLSTM(hparam)

        coder = ExampleCoder()
        data = np.array(map_dir(lambda fn: coder.encode(get_midi(fn)),
                                './datasets/easymusicnotes'))

        def fetch_data(batch_size):
            x, y = [], []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0] -
                                          hparam.timesteps-1)
                xi = data[ind][start:start+hparam.timesteps]
                yi = data[ind][start+1:start+hparam.timesteps+1]
                x.append(coder.onehot(xi))
                y.append(yi)
            x, y = np.array(x), np.array(y)
            return x, y

        model.train(fetch_data)

    elif mode == 'generate':
        hparam = StackedLSTMHParam(batch_size=1,
                                   cells=[256, 256],
                                   input_dim=363,
                                   timesteps=100,
                                   output_dim=363,
                                   temperature=1.0)
        coder = ExampleCoder()
        model = StackedLSTM(hparam)

        def handle(output):
            ind = np.random.choice(hparam.output_dim,
                                   p=output.flatten())
            res = np.zeros_like(output)
            res[0, 0, ind] = 1
            return res

        GEN_LEN = 1000
        result = model.generate(length=GEN_LEN,
                                handle=handle)
        result = result.reshape(GEN_LEN,
                                hparam.output_dim)
        result = coder.decode(result)

        save_midi('example.mid', result)
