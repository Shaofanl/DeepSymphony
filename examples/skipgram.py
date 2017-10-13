import numpy as np
from keras.preprocessing.sequence import skipgrams

from DeepSymphony.models import EmbeddingModel
from DeepSymphony.coders import AllInOneCoder
from time import sleep

from sklearn import manifold, decomposition
import matplotlib.pyplot as plt


# from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits
def plot_embedding(X, labels, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    color = ['red']*128+['blue']*128+['green']*100+['black']*7
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=color[i])
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=color[i],
                 fontdict={'weight': 'bold', 'size': 9})


if __name__ == '__main__':
    DIM = 128+128+100+7
    LEN = 100
    BS = 32

    data = np.load('./datasets/e-comp-allinone.npz')['data']

    # sample training data
    def data_generator():
        while True:
            x = []
            y = []
            for _ in range(BS):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0]-LEN-1)
                seq = data[ind][start:start+LEN]

                # #pairs ~ 2*ws*(1+neg)*seq
                pairs, labels = skipgrams(seq, DIM,
                                          window_size=50, negative_samples=1.,
                                          shuffle=True, categorical=False,
                                          sampling_table=None)
                x.append(np.array(pairs))
                y.append(np.array(labels))
            x = np.array(x)
            y = np.array(y)
            yield ([x[:, :, 0], x[:, :, 1]], y)

    (x1, x2), sim = data_generator().next()

    model = EmbeddingModel(input_dim=DIM,
                           output_dim=512,
                           seq_len=x1.shape[1])

#   model.train(data_generator(),
#               save_path='/tmp/emb.h5',
#               steps_per_epoch=50,
#               epochs=1000)

    coder = AllInOneCoder()
    x = np.arange(DIM).reshape(-1, 1)
    gen = model.build_generator('temp/emb.h5')
    res = model.generate(x).reshape(DIM, -1)

    pca = decomposition.PCA(n_components=2, )
    tsne = manifold.TSNE(n_components=2, )
    plot_embedding(tsne.fit_transform(res),
                   map(coder.code_to_name, x.flatten()))
    plt.show()

    print res.shape
#   while True:
#       ind = np.random.randint(DIM)
    for ind in range(DIM):
        print('Most similar notes with {}'.format(coder.code_to_name(ind)))
        this = res[ind]

        # dist = (((this-res)**2).sum(1)**.5)
        # rank = dist.argsort()
        sim = res.dot(this)
        rank = sim.argsort()[::-1]
        print('\t{}'.format(', '.join(map(coder.code_to_name, rank[1:6]))))
        sleep(0)
