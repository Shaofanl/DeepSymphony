from utils import getAbsT, getHots, compose
from mido import MidiFile, MidiTrack
import os
import numpy as np

from keras.models import load_model


if __name__ == '__main__':
    FILE = 'datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid'
    SONG_LEN = 400
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
    notes = [hots[i] for i in range(20)]
    accumulate = np.zeros((dim,)).astype('int')
    for _ in range(SONG_LEN):
        note = model.predict(np.array([seq]))[0]
        seq.pop(0)

        print ''.join([('x' if char >= THRESHOLD else '_') for char in note])
        notes.append(note)

        # use output as input 
#       note = note * np.random.uniform(0.75, 1.5, size=(dim, ))
        seq.append(note)

    compose(track, np.array(notes), deltat=100, threshold=THRESHOLD)
    mid.save('memorize.mid')


