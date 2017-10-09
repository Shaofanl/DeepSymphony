import numpy as np
from keras.preprocessing.sequence import skipgrams

from DeepSymphony.models import EmbeddingModel
from DeepSymphony.coders import AllInOneCoder
from time import sleep


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
