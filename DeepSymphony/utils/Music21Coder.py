import numpy as np
import music21 as ms


class NoteDurationCoder(object):
    '''
        code = onehot vector of notes + onehot vector of duration
    '''
    def __init__(self,
                 keys=128,
                 resolution=0.25,
                 maxduration=16,
                 normalize_key=None,  # the normalize target
                 single=False,  # randomly pick one from chord
                 first_voice=False):  # only return the first voice
        self.keys = keys
        self.resolution = resolution
        self.maxduration = maxduration
        self.normalize_key = normalize_key
        self.single = single
        # drepcated
        self.first_voice = first_voice

    def encode(self, score,):
        if self.first_voice:
            raise NotImplemented("buged")

        notes = []  # (offset, pitch, duration)
        for part in score.parts:
            if part.partName is None or 'Piano' not in part.partName:
                continue

            nb_voice = len([e for e in part.voices])
            if nb_voice == 0:
                continue

            print 'number of voices', nb_voice
            if self.first_voice:
                part = part.voices[0]
            print 'key', part.analyze('key')

            part = part.flat
            if self.normalize_key:
                pitches = part.analyze('key').pitches
                interval = ms.interval.Interval(
                    pitches[0], ms.pitch.Pitch(self.normalize_key)
                )
                part = part.transpose(interval)
                print 'transpose key to', part.analyze('key')

            for comp in part.notes:
                start = int(comp.offset/self.resolution)
                duration = max(int(comp.quarterLength/self.resolution), 1)
                if isinstance(comp, ms.note.Note):
                    notes.append((start, comp.pitch.midi, duration))
                elif isinstance(comp, ms.chord.Chord):
                    if self.single:
                        c = np.random.choice(comp)
                        notes.append((start, c.pitch.midi, duration))
                    else:
                        for c in comp:
                            notes.append((start, c.pitch.midi, duration))
        if len(notes) == 0:
            return [], []

        notes = sorted(notes)
        # codes:
        #   notecode = token of keys and skip
        #   duracode = token of maxduration
        notecode = []
        duracode = []
        last_t = notes[0][0]
        for ind, note in enumerate(notes):
            if note[0] > last_t:
                # skip action
                notecode.append(self.keys)
                duracode.append(note[0]-last_t)
                last_t = note[0]
            notecode.append(note[1])
            duracode.append(note[2])
        duracode = map(lambda x: min(x, self.maxduration), duracode)

        assert(min(duracode) > 0 and max(duracode) <= self.maxduration)
        assert(max(notecode) <= self.keys)
        return np.array(notecode), np.array(duracode)

    def decode(self, notecode, duracode):
        if isinstance(duracode, int):
            duracode = [duracode] * len(notecode)
        s = ms.stream.Stream()

        current_t = 0
        for note, dura in zip(notecode, duracode):
            if note == self.keys:
                current_t += dura*self.resolution
            else:
                duration = ms.duration.Duration(dura*self.resolution)
                s.insert(current_t,
                         ms.note.Note(note,
                                      duration=duration))
        return s


class MultiHotCoder(object):
    def __init__(self, keys=128, resolution=0.25):
        self.keys = keys
        self.resolution = resolution

    def encode(self, score,):
        length = int(score.duration.quarterLength/self.resolution)

        voices = []
        for stream in score.parts:
            if 'Piano' not in stream.partName:
                continue

            for comp in stream:
                # if isinstance(comp, ms.instrument.Instrument):
                #     if not isinstance(comp, ms.instrument.Piano):
                #         # an instrument other than piano
                #         break

                if isinstance(comp, ms.stream.Voice):
                    voice = np.zeros((length, self.keys))
                    for note in comp:
                        start = int(note.offset/self.resolution)
                        len = int(note.duration.quarterLength/self.resolution)
                        if isinstance(note, ms.note.Note):
                            voice[start:start+len, note.pitch.midi] = 1.0
                        elif isinstance(note, ms.chord.Chord):
                            for c in note:
                                voice[start:start+len, c.pitch.midi] = 1.0
                    voices.append(voice)
        voices = np.array(voices)
        return voices

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


class MeasureSplitCoder(NoteDurationCoder):
    def __init__(self,
                 keys=128,
                 resolution=0.25,
                 normalize_key=None,  # the normalize target
                 ):
        self.keys = keys
        self.normalize_key = normalize_key
        self.resolution = resolution

    def encode(self, score,):
        # Score > Part > Measure > Voice

        if self.normalize_key:
            cur_key = score.analyze('key').pitches
            interval = ms.interval.Interval(
                cur_key[0], ms.pitch.Pitch(self.normalize_key)
            )
            score = score.transpose(interval)
            print 'transpose key to', score.analyze('key')

        notes = []
        duras = []
        for part in score.measures(0, None):
            if part.partName != 'Piano':
                continue

            for measure in part:
                if isinstance(measure, ms.instrument.Instrument):
                    continue
                measure_res = []
                for voice in measure:
                    if isinstance(voice, ms.clef.Clef):
                        continue
                    if isinstance(voice, ms.meter.TimeSignature):
                        continue
                    if isinstance(voice, ms.bar.Barline):
                        continue

                    for ele in voice:
                        if not hasattr(ele, '__iter__'):
                            ele = [ele]  # standarize note and chord
                        for c in ele:
                            if isinstance(c, ms.note.Rest):
                                continue
                            start = int(np.round(c.offset/self.resolution))
                            duration = max(int(np.round(
                                c.quarterLength/self.resolution)), 1)
                            measure_res.append((start, c.pitch.midi, duration))

                    measure_res = sorted(measure_res)
                    note, dura = self._serialize(measure_res)
                    notes.append(note)
                    duras.append(dura)
        notes = np.array(notes)
        duras = np.array(duras)
        return notes, duras

    def _serialize(self, notes):
        notecode = []
        duracode = []
        last_t = 0
        for ind, note in enumerate(notes):
            if note[0] > last_t:
                # skip action
                notecode.append(self.keys)
                duracode.append(note[0]-last_t)
                last_t = note[0]
            notecode.append(note[1])
            duracode.append(note[2])
        # duracode = map(lambda x: min(x, self.maxduration), duracode)
        return np.array(notecode), np.array(duracode)

    def decode(self, notecode, duracode):
        if isinstance(duracode, int):
            duracode = [duracode] * len(notecode)
        s = ms.stream.Stream()

        current_t = 0
        for note, dura in zip(notecode, duracode):
            if note == self.keys:
                current_t += dura*self.resolution
            else:
                duration = ms.duration.Duration(dura*self.resolution)
                s.insert(current_t,
                         ms.note.Note(note,
                                      duration=duration))
        return s


if __name__ == '__main__':
    # coder = MultiHotCoder()
    # coder = NoteDurationCoder()
    coder = MeasureSplitCoder()
    res = coder.encode(ms.converter.parse('/home/ly/projects/deepsymphony/datasets/easymusicnotes/level11/the-penny-theme-victor-m-barba-movies-piano-level-11.mid'))
    print res

