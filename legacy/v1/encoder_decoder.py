import numpy as np
from mido import Message
import abc


class EncoderBase(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        raise NotImplemented

    @abc.abstractmethod
    def encode(self, seq, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def __call__(self, seq, **inputs):
        return self.encode(seq, **inputs)

    @abc.abstractmethod
    def decode(self, seq, **kwargs):
        raise NotImplemented

    @abc.abstractmethod
    def event_to_code(self, event):
        raise NotImplemented


class OneHotEncoder(EncoderBase):
    def __init__(self,
                 quantify_time_kwargs={},
                 hots_kwargs={}):
        self.quantify_time_kwargs = quantify_time_kwargs
        self.hots_kwargs = hots_kwargs

    def encode(self, seq):
        msgs, times = get_absolute_time(seq, **self.quantify_time_kwargs)
        hots = get_hots(msgs, times, **self.hots_kwargs)
        return hots


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

        check http://mido.readthedocs.io/en/latest/midi_files.html?highlight=tick2second

        Minutes |                               |
        --------|-------------------------------| beats per minute (BPM=4)
        Beats   | x   x   x   x | x   x   x   x |
        --------|-------------------------------| ticks per beat (TPB=3)
        Ticks   |^^^|^^^|^^^|^^^|^^^|^^^|^^^|^^^| or pulses per quarter note (PPQ=3)

        60000 / (BPM * PPQ)
        (i.e. a 120 BPM track would have a MIDI time of (60000 / (120 * 192)) or 2.604 ms for 1 tick.
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


########################################################
# All In One encoder
########################################################
class AllInOneEncoder(EncoderBase):
    EVENT_RANGE = [128,  # note on
                   128,  # note off
                   100,  # shift in (10ms, 1000ms)
                   7]    # velocity in {1, 2, 4, 8, 16, 32, 64}
    EVENT_LEN = sum(EVENT_RANGE)

    def __init__(self, return_indices=False):
        self.return_indices = return_indices

    def event_to_code(self, event, prefix=0):
        event = event + sum(self.EVENT_RANGE[:prefix])
        if self.return_indices:
            return event
        else:
            codei = np.zeros((self.EVENT_LEN,), dtype='bool')
            codei[event] = 1
            return codei

    def encode(self, seq):
        codes = []
        current_velocity = -1
        for msg in seq:
            # time event
            if msg.time > 0:
                delta = msg.time
                while delta >= 1.0:
                    delta -= 1.0
                    codes.append(
                        self.event_to_code(-1, prefix=2)
                    )
                if delta > 0:
                    codes.append(
                        self.event_to_code(int(delta*100), prefix=2)
                    )

            if msg.type not in ['note_on', 'note_off']:
                continue

            # velocity
            if msg.velocity > 0:
                velocity = int(np.log2(msg.velocity))
                if velocity != current_velocity:
                    codes.append(self.event_to_code(velocity, prefix=3))
                    current_velocity = velocity

            # note event
            if msg.type == 'note_off' or\
                    (msg.type == 'note_on' and msg.velocity == 0):
                codes.append(self.event_to_code(msg.note, prefix=1))
            elif msg.type == 'note_on':
                codes.append(self.event_to_code(msg.note))
        return np.array(codes)

    def decode(self, codes, _MIDO_TIME_SCALE=0.8):
        msgs = []
        current_velocity = 0
        current_delay = 0.
        for code in codes:
            ind = code.argmax()
            if ind < self.EVENT_RANGE[0]:
                msgs.append(Message('note_on',
                                    note=ind,
                                    velocity=current_velocity,
                                    time=int(current_delay/_MIDO_TIME_SCALE)))
                current_delay = 0.
            elif ind < sum(self.EVENT_RANGE[:2]):
                msgs.append(Message('note_off',
                                    note=ind-self.EVENT_RANGE[0],
                                    velocity=current_velocity,
                                    time=int(current_delay/_MIDO_TIME_SCALE)))
                current_delay = 0.
            elif ind < sum(self.EVENT_RANGE[:3]):
                current_delay += (ind-sum(self.EVENT_RANGE[:2])) * 10.
            elif ind < sum(self.EVENT_RANGE[:4]):
                current_velocity = 2**(ind-sum(self.EVENT_RANGE[:3]))
            else:
                raise Exception("{} Out of coding range.".format(ind))
        return msgs


# testing
if __name__ == '__main__':
    from mido import MidiFile
    midi = MidiFile('./datasets/e-comp/2002chan01.mid')
    encoder = AllInOneEncoder()
    code = encoder.encode(midi)
    msg = encoder.decode(code)

    print code.dtype
    print code.shape
    # print msg[-10:]

    from midiwrapper import Song
    s = Song()
    track = s.add_track()
    for msgi in msg:
        track.append(msgi)
    s.save_as("decode.mid")

#   import os
#   from tqdm import tqdm
#   dirpath = './datasets/e-comp/'
#   tmppath = './datasets/e-comp-allinone/'
#   encoder = AllInOneEncoder()
#   filelist = []
#   for root, _, files in os.walk(dirpath):
#       for name in files:
#           filelist.append(os.path.join(root, name))
#   for filename in tqdm(filelist):
#       midi = Song(filename)
#       hots = encoder(midi.midi).astype('bool')
#       np.savez_compressed
#       (os.path.join(tmppath, filename.split('/')[-1]+'.npz'),
#                           hots=hots)
