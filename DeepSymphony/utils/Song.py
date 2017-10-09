from mido import MidiFile, Message, MidiTrack, MetaMessage
import numpy as np
import os
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool


class Song(object):
    def __init__(self, filename=None):
        self.midi = MidiFile(filename)

    def add_track(self):
        track = MidiTrack()
        self.midi.tracks.append(track)
        return track

    def save_as(self, filename):
        self.midi.save(filename)

    # ====================================
    #       Visualization functions
    # ====================================
    @staticmethod
    def grid_vis_songs(songs, gh=5, gw=5, margin=3):
        _, w, h = songs.shape
        vis = np.ones((gh*(h+margin), gw*(w+margin)))
        for i in range(gh):
            for j in range(gw):
                vis[i*(h+margin):i*(h+margin)+h,
                    j*(w+margin):j*(w+margin)+w] = songs[i*gw+j].T
        return vis

    # ====================================
    #       Utility functions
    # ====================================
    @staticmethod
    def load_from_dir(dirpath, encoder,
                      multiprocessing=False, **kwargs):
        filelist = []
        for root, _, files in os.walk(dirpath):
            for name in files:
                filelist.append(os.path.join(root, name))
        data = []
        if multiprocessing:
            def handle(filename):
                midi = Song(filename)
                hots = encoder(midi.midi)
                return hots
            pool = Pool(10)
            data = np.array(pool.map(handle, filelist))
        else:
            for filename in tqdm(filelist):
                try:
                    midi = Song(filename)
                    hots = encoder(midi.midi)
                    data.append(hots)
                except:
                    print "error with {}".format(filename)
            data = np.array(data)
        return data

    @staticmethod
    def _copy(source, track, filter_f=lambda x: True, coef=1000):
        """
            Copy the notes from source to target track

            source: source midfile
            target: target track
            filter_list: filter for msg to be copied, copy all msg
                if it equals to None
            coef:   coefficient of time (double -> int)
        """
        for msg in source:
            if filter_f(msg):
                track.append(msg.copy(time=int(msg.time*coef)))

    @staticmethod
    def _compose(track, notes, deltat=461.09, velocity=48, threshold=1.):
        """
            From notes to track
        """
        LEN, dim = notes.shape
        track.append(MetaMessage('set_tempo', tempo=500000))

        # notes to abs time list
        times = []
        actions = []
        T = 0
        for ind, line in enumerate(notes):
            for note in range(dim):
                if notes[ind, note] >= threshold and \
                        (ind == 0 or notes[ind-1, note] < threshold):
                    times.append(T)
                    actions.append(('note_on', note))

            T += int(deltat)

            for note in range(dim):
                if (notes[ind, note] < threshold and
                        notes[ind-1, note] >= threshold) or \
                        (ind == LEN-1 and notes[ind, note] >= threshold):
                    times.append(T)
                    actions.append(('note_off', note))

        for i in range(len(times)-1, 0, -1):
            times[i] = times[i] - times[i-1]

        for t, a in zip(times, actions):
            if a[0] == 'note_on':
                track.append(Message('note_on',
                                     note=a[1],
                                     velocity=velocity,
                                     time=t))
            else:
                track.append(Message('note_off',
                                     note=a[1],
                                     velocity=0,
                                     time=t))
