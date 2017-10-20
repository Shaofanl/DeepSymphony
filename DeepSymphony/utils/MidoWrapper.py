from mido import MidiFile, MidiTrack


def get_midi(filename):
    return MidiFile(filename)


def save_midi(filename, notes):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)
    for note in notes:
        track.append(note)
    midi.save(filename)
