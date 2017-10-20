import numpy as np

from DeepSymphony.models.StackedLSTM import (
    StackedLSTM, StackedLSTMHParam)


if __name__ == '__main__':
    hparam = StackedLSTMHParam(batch_size=30,
                               input_dim=363,
                               timesteps=100,
                               output_dim=363)
    model = StackedLSTM(hparam)

    def fetch_data(batch_size):
        x = np.random.rand(hparam.batch_size,
                           hparam.timesteps,
                           hparam.input_dim)
        x /= x.sum(2)[:, :, None]
        y = np.random.randint(hparam.output_dim,
                              size=(hparam.batch_size,
                                    hparam.timesteps))
        return x, y

    model.train(fetch_data)
