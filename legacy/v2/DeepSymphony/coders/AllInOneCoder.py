########################################################
# All In One encoder
########################################################
from .CoderBase import CoderBase
from ..utils.constants import get_note_name
import numpy as np
from mido import Message


class AllInOneCoder(CoderBase):
    EVENT_RANGE = [128,  # note on
                   128,  # note off
                   100,  # shift in (10ms, 1000ms)
                   7]    # velocity in {1, 2, 4, 8, 16, 32, 64}
    EVENT_LEN = sum(EVENT_RANGE)

    def __init__(self, return_indices=False):
        self.return_indices = return_indices

    def code_to_name(self, ind):
        assert 0 <= ind <= self.EVENT_LEN

        if ind < self.EVENT_RANGE[0]:
            return '<'+get_note_name(ind)+' on>'
        elif ind < sum(self.EVENT_RANGE[:2]):
            return '<'+get_note_name(ind-self.EVENT_RANGE[0])+' off>'
        elif ind < sum(self.EVENT_RANGE[:3]):
            T = (ind-sum(self.EVENT_RANGE[:2])) * 10.
            return '<delay {}>'.format(T)
        elif ind < sum(self.EVENT_RANGE[:4]):
            V = 2**(ind-sum(self.EVENT_RANGE[:3]))
            return '<velocity {}>'.format(V)

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
                    codes.append(self.event_to_code(-1, prefix=2))
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

    def decode(self, codes, _MIDO_TIME_SCALE=0.8, **kwargs):
        # post process
        last_appear = np.ones((128,)) * (-1)
        post_process = []
        current_t = 0.
        for note in codes:
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
                   current_t - last_appear[key] > kwargs.get('max_sustain', 2.0):
                    # print('force disable {}'.format(key))
                    stop = np.zeros((363,))
                    stop[key+128] = 1.
                    last_appear[key] = -1
                    post_process.append(stop)
        codes = post_process

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
