import numpy as np
from mido import Message


class ExampleCoder(object):
    def __init__(self, return_onehot=False, shift_count=100):
        self.return_onehot = return_onehot
        self.EVENT_RANGE = [128,  # note on
                            128,  # note off
                            shift_count,  # shift in (10ms, 1000ms)
                            7]    # velocity in {1, 2, 4, 8, 16, 32, 64}
        self.EVENT_COUNT = sum(self.EVENT_RANGE)
        self.shift_count = shift_count

    def event_to_code(self, event, prefix=0):
        event = int(event) + sum(self.EVENT_RANGE[:prefix])
        return event

    def onehot(self, seq):
        return np.eye(self.EVENT_COUNT)[seq]

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
                        self.event_to_code(
                            np.clip(np.round(delta*self.shift_count),
                                    1, self.shift_count),
                            prefix=2)
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

        codes = np.array(codes)
        if self.return_onehot:
            return self.onehot(codes)
        return codes

    def decode(self, codes, _MIDO_TIME_SCALE=1.0, **kwargs):
        # post process
        last_appear = np.ones((128,)) * (-1)
        post_process = []
        current_t = 0.
        for note in codes:
            post_process.append(note)
            if not isinstance(note, int):
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
                   current_t - last_appear[key] > kwargs.get('max_sustain',
                                                             2.0):
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
            if isinstance(code, int):
                ind = code
            else:
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
                current_delay += (ind-sum(self.EVENT_RANGE[:2])) * \
                        1000./self.shift_count
            elif ind < sum(self.EVENT_RANGE[:4]):
                current_velocity = 2**(ind-sum(self.EVENT_RANGE[:3]))
            else:
                raise Exception("{} Out of coding range.".format(ind))
        return msgs
