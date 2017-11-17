import mido
import numpy as np
from mido import Message
from time import sleep


class Player(object):
    '''
        First use `timidity -iA` to open a daemon program,
            which can synthesize the message into music.
        Recommend: use pianoteq
    '''
    def __init__(self,
                 outport_name='TiMidity:TiMidity port 0 129:0'):
        self.port = mido.open_output(outport_name)

    def play(self, message):
        self.port.send(message)


class ExampleCoderPlayer(Player):
    def __init__(self, decoder, speed=1.0, max_sustain=2.0, **kwargs):
        self.decoder = decoder
        self.current_t = 0  # abs t
        self.current_d = 0  # delay
        self.current_v = 32
        self.last_appear = np.ones((128,)) * (-1)
        self.max_sustain = max_sustain
        self.speed = speed

        super(ExampleCoderPlayer, self).__init__(**kwargs)

    def play(self, ind):
        decoder = self.decoder
        for key in range(128):
            if self.last_appear[key] > 0 and \
               self.current_t - self.last_appear[key] > self.max_sustain:
                msg = Message('note_off', note=key)
                self.last_appear[key] = -1
                self.port.send(msg)
                self.port.send(msg)

        if ind < decoder.EVENT_RANGE[0]:
            if self.last_appear[ind] == -1:
                self.last_appear[ind] = self.current_t
            sleep(self.current_d/self.speed)
            msg = Message('note_on',
                          note=ind,
                          velocity=self.current_v)
            self.current_d = 0.
            self.port.send(msg)
        elif ind < sum(decoder.EVENT_RANGE[:2]):
            self.last_appear[ind-128] = -1
            sleep(self.current_d/self.speed)
            msg = Message('note_off',
                          note=ind-decoder.EVENT_RANGE[0])
            self.current_d = 0.
            self.port.send(msg)
        elif ind < sum(decoder.EVENT_RANGE[:3]):
            delta = (ind-sum(decoder.EVENT_RANGE[:2])) * \
                    1./decoder.shift_count
            self.current_d += delta
            self.current_t += delta
        elif ind < sum(decoder.EVENT_RANGE[:4]):
            self.current_v = 2**(ind-sum(decoder.EVENT_RANGE[:3]))


class MultihotsPlayer(Player):
    def __init__(self, threshold, speed, **kwargs):
        self.keyboard = np.zeros((128,))
        self.threshold = threshold
        self.current_v = 60
        self.speed = speed

        super(MultihotsPlayer, self).__init__(**kwargs)

    def play(self, hots):
        for ind, ele in enumerate(hots):
            if self.keyboard[ind] == 0:
                if ele > self.threshold:
                    msg = Message('note_on',
                                  note=ind,
                                  velocity=self.current_v)
                    self.port.send(msg)
                    self.keyboard[ind] = 1
            else:
                if ele <= self.threshold:
                    self.keyboard[ind] = 0
                    msg = Message('note_off',
                                  note=ind,
                                  velocity=self.current_v)
                    self.port.send(msg)
        sleep(1./self.speed)
