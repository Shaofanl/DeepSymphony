from keras.models import load_model
from keras.layers import LSTM
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import compose

if __name__ == '__main__':
    SONG_LEN = 500
    THRESHOLD = 0.50
    MAX_SUSTAIN = 4

    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    model = load_model('temp/simple_rnn.h5')
    _, LEN, dim = model.input_shape

    def get_random():
#       return [np.random.rand(dim,) for _ in range(LEN)]
        return [np.random.binomial(1, 0.5, (dim,)) for _ in range(LEN)]

    # using random as openning 
#   seq = [np.random.rand(dim,) for _ in range(LEN)]
    seq = get_random() 
    notes = []
    accumulate = np.zeros((dim,)).astype('int')
    for _ in range(SONG_LEN):
        note = model.predict(np.array([seq]))[0]
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
            #note /= note.max()
            #note = np.random.rand(dim,)
            seq = get_random() 
        else:
            # use output as input 
            note = note * np.random.uniform(0.5, 2.0, size=(dim, ))
            seq.append(note)

    compose(track, np.array(notes), deltat=200, threshold=THRESHOLD)
    mid.save('simple_rnn.mid')
