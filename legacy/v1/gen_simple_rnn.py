from keras.models import load_model, Model
import numpy as np
from midiwrapper import Song
from encoder_decoder import AllInOneEncoder
from model_simple_rnn import define_model


def get_openning(LEN, mode='borrow'):
    if mode == 'borrow':
        # borrow from song
        # midi = MidiFile("songs/bach_846.mid")
        midi = Song('./datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid')
        # midi = Song('./datasets/e-comp/2002chan01.mid')
        hots = AllInOneEncoder().encode(midi.midi)
        # hots = midi.encode_onehot()
        return [hots[i] for i in range(LEN)]
    elif mode == 'random':
        # random
        # return [np.random.rand(dim,) for _ in range(LEN)]
        return [np.random.binomial(1, 0.2, (dim,)) for _ in range(LEN)]
    else:
        raise NotImplemented


if __name__ == '__main__':
    SONG_LEN = 6000
    THRESHOLD = 1.00

    mid = Song()
    track = mid.add_track()

    # model = load_model('temp/simple_rnn.h5', compile=False)
    model = define_model((1, 1, 363), stateful=True)
    model.load_weights('temp/simple_rnn.h5')
    model = Model(model.input, model.layers[-2].output)  # before softmax
    _, LEN, dim = model.input_shape

    # avoid too long sustain
    sustain = np.zeros((128))
    current_t = 0.

    np.random.seed( sum(map(ord, 'wuxintong')) )  # 32)

    seq = get_openning(LEN, mode='random')
    for seqi in seq:
        res = model.predict(np.array([[seqi]]))
    res = model.predict(np.zeros((1, 1, 363)))
    notes = []  # deepcopy(seq)
    temperature = 1.0
    for _ in range(SONG_LEN):
        note = res[0][-1]
        note = np.exp(note/temperature)
        note /= note.sum()

        ind = np.random.choice(len(note), p=note)
        note = np.zeros_like(note)
        note[ind] = 1
        res = model.predict(np.array([[note]]))

        print ''.join([('x' if char >= THRESHOLD else '_') for char in note[:128]])
        print ''.join([('x' if char >= THRESHOLD else '_') for char in note[128:256]])
        print ''.join([('x' if char >= THRESHOLD else '_') for char in note[256:356]])
        print ''.join([('x' if char >= THRESHOLD else '_') for char in note[356:]])
        notes.append(note)

    # handle
    MAX_SUSTAIN = 2.0
    last_appear = np.ones((128,)) * (-1)
    post_process = []
    current_t = 0.
    for note in notes:
        post_process.append(note)
        note = note.argmax()
        # print note
        if note < 128:
            if last_appear[note] == -1:
                last_appear[note] = current_t
        elif note < 256:
            last_appear[note-128] = -1
        elif note < 356:
            current_t += (note-256)*0.1

        for key in range(128):
            if last_appear[key] > 0 and \
               current_t - last_appear[key] > MAX_SUSTAIN:
                # print('force disable {}'.format(key))
                stop = np.zeros((363,))
                stop[key+128] = 1.
                last_appear[key] = -1
                post_process.append(stop)
        # print last_appear

    # mid.compose(track, np.array(notes), deltat=200, threshold=THRESHOLD)
    # , _MIDO_TIME_SCALE=0.6):
    for msgi in AllInOneEncoder().decode(post_process):
        track.append(msgi)
    mid.save_as('simple_rnn.mid')
