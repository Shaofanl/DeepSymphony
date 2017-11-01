import numpy as np


def continuous_sample(seq, window_size, stride=1):
    res = []
    i = 0
    for i in range(0, len(seq)-window_size, stride):
        res.append(seq[i:i+window_size])
    return np.array(res)
