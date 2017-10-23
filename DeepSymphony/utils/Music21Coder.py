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

    def decode(self, codes):
        # TODO: check correctness
        nb_voices = codes.shape[0]
        stream = ms.stream.Stream()
        for vind in range(nb_voices):
            starts = np.ones((codes.shape[-1],), dtype='int')*(-1)
            voice = ms.stream.Voice()

            codei = codes[vind]
            codei = np.vstack([codei, np.zeros((codei.shape[1],))])
            print codei.shape
            for tind in range(codei.shape[0]):
                for nind in range(codei.shape[1]):
                    if codei[tind, nind] > 0:
                        if starts[nind] == -1:
                            starts[nind] = tind
                    else:
                        if starts[nind] != -1:
                            note = ms.note.Note(nind)
                            note.duration = \
                                ms.duration.Duration(tind-starts[nind])
                            voice.insert(starts[nind], note)
                            starts[nind] = -1
            stream.append(voice)
        return stream
