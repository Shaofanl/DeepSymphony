from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import copy

mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

source = MidiFile('playground/AutumnL.mid')
# only copy notes
copy(source, track, filter_f=lambda x: x.type == 'note_on')
mid.save('playground/AutumnL_copy.mid')