from keras.models import load_model
from keras.layers import LSTM
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import compose

if __name__ == '__main__':
    SONG_LEN = 500
    THRESHOLD = 0.30
    MAX_SUSTAIN = 8

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    model = load_model('temp/simple_rnn.h5')
    _, LEN, dim = model.input_shape

    def get_random():
#       return [np.random.rand(dim,) for _ in range(LEN)]
        return [np.random.binomial(1, 0.8, (dim,)) for _ in range(LEN)]

    # using random as openning 
#   seq = [np.random.rand(dim,) for _ in range(LEN)]
    seq = get_random() 
    notes = []
    accumulate = np.zeros((dim,))
    for _ in range(SONG_LEN):
        note = model.predict(np.array([seq]))[0]
        seq.pop(0)
        seq.append(note)
        print ''.join([('x' if char >= THRESHOLD else '_') for char in note])

        # sustain too long
        accumulate += (note>= THRESHOLD)
        note[accumulate>=MAX_SUSTAIN] = 0.
        accumulate[accumulate>=MAX_SUSTAIN] = 0.

        # too less notes
        if (note>= THRESHOLD).sum() == 0:
            print 'no note, max =', note.max()
            #note /= note.max()
            #note = np.random.rand(dim,)
            seq = get_random() 

        # noise
        note = note * np.random.uniform(0.8, 1.2, size=(dim, ))

        notes.append(note)

    compose(track, np.array(notes), deltat=200, threshold=THRESHOLD)
    mid.save('simple_rnn.mid')
