NOTE_NUMBER = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTE_COUNT = len(NOTE_NUMBER)
OCTAVE_COUNT = 10


def get_note_name(index):
    assert 0 <= index <= 127
    return '{}{}'.format(NOTE_NUMBER[index % NOTE_COUNT], index / NOTE_COUNT)
