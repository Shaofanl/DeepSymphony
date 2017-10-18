import numpy as np

from DeepSymphony.coders import AllInOneCoder
from DeepSymphony.utils.stat import histogram
from DeepSymphony.utils.constants import NOTE_NUMBER
from sklearn import manifold
import matplotlib.pyplot as plt
# from multiprocessing import Pool


def get_hist(song):
    events = [coder.code_to_name(note) for note in song]
    hist = histogram(events, NOTE_NUMBER)
    hist /= hist.sum()
    print hist
    return hist


if __name__ == '__main__':
    DIM = 128+128+100+7

    data = np.load('./datasets/e-comp-allinone.npz')['data']
    coder = AllInOneCoder()
    np.set_printoptions(precision=2)

    # pool = Pool(8)
    # hist = np.array(pool.map(get_hist, data))
    # np.savez('examples/hist.npz', hist=hist)
    hist = np.load('examples/hist.npz')['hist']
    hist = np.exp(hist*100.0)
    hist /= hist.sum(1)[:, None]
    print hist[:10]

    tsne = manifold.TSNE(n_components=2, angle=0.8, verbose=1)
    hist = tsne.fit_transform(hist)
    plt.scatter(hist[:, 0], hist[:, 1])
    plt.show()
