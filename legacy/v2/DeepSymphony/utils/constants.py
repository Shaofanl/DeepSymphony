NOTE_NUMBER = ['C_', 'C#', 'D_', 'D#', 'E_',
               'F_', 'F#', 'G_', 'G#', 'A_', 'A#', 'B_']
NOTE_COUNT = len(NOTE_NUMBER)
OCTAVE_COUNT = 10


def get_note_name(index):
    assert 0 <= index <= 127
    return '{}{}'.format(NOTE_NUMBER[index % NOTE_COUNT], index / NOTE_COUNT)
