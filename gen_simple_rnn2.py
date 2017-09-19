from keras.models import load_model
from keras.layers import LSTM
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import compose, getAbsT, getHots
from copy import deepcopy

def get_openning(LEN, mode='borrow'):
    if mode == 'borrow':
        # borrow from song 
        midi = MidiFile("songs/bach_846.mid")
        msgs, times = getAbsT(midi, filter_f=lambda x: x.type in ['note_on', 'note_off'], unit='beat')
        hots = getHots(msgs, times, resolution=0.25)
        return [hots[i] for i in range(LEN)]
    elif mode == 'random':
        # random
        # return [np.random.rand(dim,) for _ in range(LEN)]
        return [np.random.binomial(1, 0.3, (dim,)) for _ in range(LEN)]
    else:
        raise NotImplemented

if __name__ == '__main__':
    SONG_LEN = 500
    THRESHOLD = 0.50
    MAX_SUSTAIN = 4

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    model = load_model('temp/simple_rnn.h5')
    _, LEN, dim = model.input_shape

    seq = get_openning(LEN, mode='borrow') 
    notes = [] #deepcopy(seq)
    accumulate = np.zeros((dim,)).astype('int')
    for _ in range(SONG_LEN):
        note = model.predict(np.array([seq]))[0][-1]
        seq.pop(0)

        # sustain too long
        accumulate = accumulate*(note>= THRESHOLD) + (note>= THRESHOLD)
        note[accumulate>=MAX_SUSTAIN] = 0.
        accumulate[accumulate>=MAX_SUSTAIN] = 0.

        print ''.join([('x' if char >= THRESHOLD else '_') for char in note])
        notes.append(note)

        if (note>= THRESHOLD).sum() == 0:
            # too less notes
            print 'no note, max =', note.max()
            #note = np.random.rand(dim,)
			#seq = get_openning(LEN)
            if note.max() > 0: note /= note.max()
            seq.append(note)
        else:
            # use output as input 
            note = note * np.random.uniform(0.5, 2.0, size=(dim, ))
            seq.append(note)

    compose(track, np.array(notes), deltat=200, threshold=THRESHOLD)
    mid.save('simple_rnn.mid')



