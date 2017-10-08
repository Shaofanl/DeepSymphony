import numpy as np
from keras.preprocessing.sequence import skipgrams


if __name__ == '__main__':
    LEN = 2000  # length of input
    DIM = 128+128+100+7

    data = np.load('./datasets/e-comp-allinone.npz')['data']

    import ipdb
    ipdb.set_trace()

    skipgrams(sequence, vocabulary_size,
              window_size=4, negative_samples=1., shuffle=True,
              categorical=False, sampling_table=None)
