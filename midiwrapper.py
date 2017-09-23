from mido import MidiFile, Message, MidiTrack, MetaMessage
import numpy as np
import mido


class Song(object):
    def __init__(self, filename=None):
        self.midi = MidiFile(filename)

    def add_track(self):
        track = MidiTrack()
        self.midi.tracks.append(track)
        return track

    def save_as(self, filename):
        self.midi.save(filename)


    #====================================
    #       Encoding functions
    #====================================
    def encode_onehot(self,
            kwargs1={'filter_f':lambda x: x.type in ['note_on', 'note_off'], 'unit':'beat'},
            kwargs2={'resolution':0.25}):
        msgs, times = self._get_absolute_time(self.midi, **kwargs1)
        hots = self._get_hots(msgs, times, **kwargs2)
        return hots


    #====================================
    #       Utility functions 
    #====================================
    def get_absolute_time(self, **kwargs):
        return self._get_absolute_time(self.midi, **kwargs)

    @staticmethod
    def _get_absolute_time(source, filter_f=lambda x: True, unit='second', quantize=4):
        """
            Translate a relative time-format into an
                absolute time-format.

            source: source MIDI object
            filter: filter of notes
            beat: turn time into beat unit 
            quantize: quantize number to (1/Q)
                (e.g. Q=4 makes (1/4*beat=) 1/16 the smallest note)

            check http://mido.readthedocs.io/en/latest/midi_files.html?highlight=tick2second

            Minutes |                               |
            --------|-------------------------------| beats per minute (BPM=4)
            Beats   | x   x   x   x | x   x   x   x | 
            --------|-------------------------------| ticks per beat (TPB=3)
            Ticks   |^^^|^^^|^^^|^^^|^^^|^^^|^^^|^^^| or pulses per quarter note (PPQ=3)

            60000 / (BPM * PPQ)
            (i.e. a 120 BPM track would have a MIDI time of (60000 / (120 * 192)) or 2.604 ms for 1 tick.
        """
        tempo = 500000 # 120BPM

        timestamps = []
        T = 0.0
        messages = []
        for msg in source:
            if msg.type == 'set_tempo':
                tempo = msg.tempo 

            # TODO: whether take in other delta-time 
            if filter_f(msg):
                if msg.time > 0: 
                    t = float(msg.time)
                    if unit=='second':
                        t = round(t*quantize)/quantize 
                    elif unit=='beat':
                        t = round(t*1e6/tempo*quantize)/quantize
                    else:
                        raise NotImplemented 
                    T += t

                messages.append(msg)
                timestamps.append(T)

        return messages, timestamps

    @staticmethod
    def _get_hots(msgs, times, hots=128, field='note', resolution=1.):
        """
            Translate a (msgs, times) pair into a T*hots matrix

            msgs: list of msg
            times: timestamps
            resolution: width of time-slice
            field: which field to extract

            usage:
            msg, times = getAbsT(source, ...)
            hots = getHots(msg, times)
        """
        
        n = times[-1]/resolution + 1
        res = np.zeros((int(n), hots))

        # TODO: use Cython to accelerate
        T = 0.0
        msg_ind = 0
        res_ind = 0
        while T < times[-1]:
            # sustain
            if field == 'note':
                res[res_ind] = res[res_ind-1]

            while msg_ind < len(msgs) and T >= times[msg_ind]:
                msg = msgs[msg_ind]
                if field == 'note':
                    if msg.type=='note_off' or msg.velocity == 0:
                        res[res_ind, msg.__dict__[field]] = 0.
                    else:
                        res[res_ind, msg.__dict__[field]] = 1.
                else:
                    raise NotImplemented 
                msg_ind += 1
            res_ind += 1
            T += resolution 
        return res

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
                if notes[ind, note] >= threshold and (ind == 0 or notes[ind-1, note] < threshold):
                    times.append(T)
                    actions.append(('note_on', note))

            T += int(deltat)

            for note in range(dim): 
                if (notes[ind, note] <threshold and notes[ind-1, note] >= threshold)\
                                      or (ind==LEN-1 and notes[ind, note]>= threshold):
                    times.append(T)
                    actions.append(('note_off', note))

        for i in range(len(times)-1, 0, -1):
            times[i] = times[i] - times[i-1]

        for t, a in zip(times, actions):
            if a[0] == 'note_on':
                track.append( Message('note_on', note=a[1], velocity=velocity, time=t)) 
            else:
                track.append( Message('note_off', note=a[1], velocity=0, time=t)) 

