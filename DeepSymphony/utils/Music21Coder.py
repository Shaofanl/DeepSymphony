import numpy as np
import music21 as ms


class MultiHotCoder(object):
    def __init__(self, bits=128):
        self.bits = bits

    def encode(self,
               score,
               resolution=0.25):
        length = int(score.duration.quarterLength/resolution)

        voices = []
        for stream in score:
            for comp in stream:
                if isinstance(comp, ms.stream.Voice):
                    voice = np.zeros((length, self.bits))
                    for note in comp:
                        start = int(note.offset/resolution)
                        len = int(note.duration.quarterLength/resolution)
                        if isinstance(note, ms.note.Note):
                            voice[start:start+len, note.pitch.midi] = 1.0
                        elif isinstance(note, ms.chord.Chord):
                            for c in note:
                                voice[start:start+len, c.pitch.midi] = 1.0
                    voices.append(voice)
        return np.array(voices)
