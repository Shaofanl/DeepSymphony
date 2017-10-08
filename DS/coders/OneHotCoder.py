from .CoderBase import CoderBase
import numpy as np


class OneHotCoder(CoderBase):
    def __init__(self,
                 quantify_time_kwargs={},
                 hots_kwargs={}):
        self.quantify_time_kwargs = quantify_time_kwargs
        self.hots_kwargs = hots_kwargs

    def encode(self, seq):
        msgs, times = self.get_absolute_time(seq, **self.quantify_time_kwargs)
        hots = self.get_hots(msgs, times, **self.hots_kwargs)
        return hots

    @staticmethod
    def get_absolute_time(
            source,
            filter_f=lambda x: True,
            unit='second',
            quantize=4):
        """
            Translate a relative time-format into an
                absolute time-format.

            source: source MIDI object
            filter: filter of notes
            beat: turn time into beat unit
            quantize: quantize number to (1/Q)
                (e.g. Q=4 makes (1/4*beat=) 1/16 the smallest note)

            check http://mido.readthedocs.io/en/latest/midi_files.html?
                                        highlight=tick2second

            Minutes |                               |
            --------|-------------------------------| beats per minute (BPM=4)
            Beats   | x   x   x   x | x   x   x   x |
            --------|-------------------------------| ticks per beat (TPB=3)
            Ticks   |^^^|^^^|^^^|^^^|^^^|^^^|^^^|^^^| or pulses per quarter
                                                        note (PPQ=3)

            60000 / (BPM * PPQ)
            (i.e. a 120 BPM track would have a MIDI time of
                (60000 / (120 * 192)) or 2.604 ms for 1 tick.
        """
        tempo = 500000  # 120BPM

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
                    if unit == 'second':
                        t = round(t*quantize)/quantize
                    elif unit == 'beat':
                        t = round(t*1e6/tempo*quantize)/quantize
                    else:
                        raise NotImplemented
                    T += t

                messages.append(msg)
                timestamps.append(T)

        return messages, timestamps

    @staticmethod
    def get_hots(msgs, times, hots=128, field='note', resolution=1.):
        """
            Translate a (msgs, times) pair into a T*hots matrix

            msgs: list of msg
            times: timestamps
            resolution: width of time-slice
            field: which field to extract

            usage:
            msg, times = getAbsT(source, ...)
            hots = getHots(msg, times) """
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
                    if msg.type == 'note_off' or msg.velocity == 0:
                        res[res_ind, msg.__dict__[field]] = 0.
                    else:
                        res[res_ind, msg.__dict__[field]] = 1.
                else:
                    raise NotImplemented
                msg_ind += 1
            res_ind += 1
            T += resolution
        return res
