from utils import getAbsT, getHots, compose
from mido import MidiFile, MidiTrack
import os
import numpy as np
from copy import deepcopy

from keras.models import load_model


if __name__ == '__main__':
    FILE = 'datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid'
    SONG_LEN = 1000
    LEN = 20
    dim = 128
    THRESHOLD = 0.85

    midi = MidiFile(FILE)
    msgs, times = getAbsT(midi, filter_f=lambda x: x.type in ['note_on', 'note_off'], unit='beat')
    hots = getHots(msgs, times, resolution=0.25)
    print hots.shape
    model = load_model('temp/memorize.h5')


    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    seq = [hots[i] for i in range(20)]
#   seq = [np.random.binomial(1, 0.3, (dim,)) for _ in range(LEN)]
    notes = [] #deepcopy(seq) 
    accumulate = np.zeros((dim,)).astype('int')
    for _ in range(SONG_LEN):
        note = model.predict(np.array([seq]))[0]
        seq.pop(0)

        print ''.join([('x' if char >= THRESHOLD else '_') for char in note])
        notes.append(note)

        # use output as input 
        # note = note * np.random.uniform(0.75, 1.2, size=(dim, ))
        seq.append(note)

    compose(track, np.array(notes), deltat=100, threshold=THRESHOLD)
    mid.save('memorize.mid')


