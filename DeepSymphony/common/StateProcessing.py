import numpy as np


def gather_states(collection, hparam):
    # collection:
    #    (samples,
    #     unknown(1),
    #     layers,
    #     types of hidden states (2 in LSTM, 1 in RNN),
    #     batch_size,
    #     number of units)

    # extract batch_size
    samples = len(collection)
    for i in range(samples):
        assert(len(collection[i]) == 1)
    new = []
    for i_layer in range(len(hparam.encoder_cells)):
        states_type = type(collection[0][0][i_layer])
        states_count = len(collection[0][0][i_layer])
        layer = []
        # LSTM==2 (cell and hidden), GRU=1
        for i_state in range(states_count):
            batches = [
                collection[i][0][i_layer][i_state]
                for i in range(samples)
            ]
            batches = np.concatenate(batches)
            assert(batches.shape[0] == samples*hparam.batch_size)
            layer.append(batches)
        layer = states_type(*layer)
        new.append(layer)

    labels = ['layer_{}'.format(i)
              for i in range(len(new))]
    transposed = [np.transpose(new[i], (1, 0, 2))
                  for i in range(len(new))]
    new = zip(labels, transposed)
    return dict(new)
