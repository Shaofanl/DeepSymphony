import numpy as np

from DeepSymphony.utils.bleu import bleu
from DeepSymphony.models import StackedRNN
from keras.utils.np_utils import to_categorical
from DeepSymphony.coders import AllInOneCoder
from keras.models import load_model
from keras.layers import Embedding


PREFIX = 5
ANSWER = 50
LEN = PREFIX+ANSWER
TEST = 100
DIM = 128+128+100+7
coder = AllInOneCoder()
RNG = np.random.RandomState(64)


def tokenize(x):
    if x.ndim == 1:
        return map(coder.code_to_name, x)
    elif x.ndim == 2:
        return map(coder.code_to_name, x.argmax(1))


def test_emb_stackrnn(data, weight='temp/emb_stackrnn.h5'):
    emb_w = load_model('temp/emb.h5').layers[2].layers[1].get_weights()
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM,
                       output_dim=DIM,
                       cells=[512, 512, 512],
                       embedding=Embedding(DIM, 512, trainable=False),
                       embedding_w=emb_w)
    model.build_generator(weight)

    moving_mean = []
    for _ in range(TEST):
        ind = RNG.randint(len(data))
        start = RNG.randint(data[ind].shape[0]-LEN-1)
        x = np.array(data[ind][start:start+LEN])

        score = bleu(model=model,
                     prior=x,
                     prefix_length=PREFIX,
                     tokenize=tokenize)
        moving_mean.append(score)
        print np.mean(moving_mean)


def test_stackrnn(data, weight='temp/simple_rnn.h5'):
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM,
                       output_dim=DIM,
                       cells=[512, 512, 512])
    model.build_generator(weight)

    moving_mean = []
    for _ in range(TEST):
        ind = RNG.randint(len(data))
        start = RNG.randint(data[ind].shape[0]-LEN-1)
        x = np.array(data[ind][start:start+LEN])
        x = to_categorical(x.flatten(), num_classes=DIM).\
            reshape(x.shape[0], DIM)

        score = bleu(model=model,
                     prior=x,
                     prefix_length=PREFIX,
                     tokenize=tokenize)
        moving_mean.append(score)
        print np.mean(moving_mean)


if __name__ == '__main__':
    data = np.load('./datasets/e-comp-allinone.npz')['data']
#   test_stackrnn(data, weight='temp/e-comp_simple_rnn.h5')  # 38
    test_stackrnn(data, weight='temp/simple_rnn.h5')  # 25
#   test_stackrnn(data, weight='temp/stackedrnn_kernel_l2.h5')  # 40
#   test_stackrnn(data, weight='temp/temp.h5')
#   test_emb_stackrnn(data)  # 28
