import numpy as np
from DeepSymphony.models import StackedRNN
from DeepSymphony.utils.constants import NOTE_NUMBER
from DeepSymphony.coders import AllInOneCoder
from scipy.stats import entropy as KL_divergence
from DeepSymphony.utils.stat import histogram, histogram_onehot,\
    min_norm, norm
from keras.utils.np_utils import to_categorical


def JS_divergence(P, Q):
    return 0.5*KL_divergence(P, Q)+0.5*KL_divergence(Q, P)


def define_stackrnn(LEN, DIM_IN, DIM_OUT):
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM_IN,
                       output_dim=DIM_OUT,
                       cells=[512, 512, 512],)
    model.build_generator('temp/e-comp_simple_rnn.h5')
    # model.build_generator('temp/temp.h5')
    return model


def define_stackrnn_hist(LEN, DIM_IN, DIM_OUT):
    model = StackedRNN(timespan=LEN,
                       input_dim=DIM_IN,
                       output_dim=DIM_OUT,
                       cells=[512, 512, 512],)
#   model.build_generator('temp/stackedrnn_hist.h5')
    model.build_generator('temp/stackedrnn_hist_patch_T=0.05.h5')
    return model


if __name__ == '__main__':
    PREFIX = 500
    ANSWER = 2000
    LEN = PREFIX+ANSWER
    DIM_IN = 128+128+100+7
    DIM_OUT = 128+128+100+7
    TEST = 1000
    RNG = np.random.RandomState(64)
    TEMPERATURE = 0.05

    HIST = False
    # HIST = True

    data = np.load('./datasets/e-comp-allinone.npz')['data']
    hist = np.load('./datasets/e-comp-allinone-hist.npz')['hist']
    np.set_printoptions(precision=2)
    coder = AllInOneCoder()

    if HIST:
        model = define_stackrnn_hist(LEN, DIM_IN+len(NOTE_NUMBER), DIM_OUT)
    else:
        model = define_stackrnn(LEN, DIM_IN, DIM_OUT)

    D = []
    for _ in range(TEST):
        ind = RNG.randint(len(data))
        start = RNG.randint(data[ind].shape[0]-LEN-1)
        x = np.array(data[ind][start:start+LEN])
        h_pre = histogram_onehot(x[:PREFIX], coder.code_to_name, NOTE_NUMBER)
        h_ans = histogram_onehot(x[-ANSWER:], coder.code_to_name, NOTE_NUMBER)
        x = to_categorical(x[:PREFIX], num_classes=DIM_IN).\
            reshape(PREFIX, DIM_IN)

        model.reset_generator()
        song = model.generate(seed=RNG.randint(1e+9),
                              length=ANSWER,
                              prefix=x[:PREFIX],
                              verbose=0,
                              addition=norm(h_ans, TEMPERATURE)
                              if HIST else None)

        hist_res = histogram_onehot(song, coder.code_to_name, NOTE_NUMBER)
        d = JS_divergence(h_ans, hist_res)

        print h_ans
        print hist_res
        print d

        if d != np.inf:
            D.append(d)
        print np.mean(D)
