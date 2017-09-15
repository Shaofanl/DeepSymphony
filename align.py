from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import getAbsT
from itertools import izip
from time import sleep

source = MidiFile('songs/anniversary-song.mid')
msgs, times = getAbsT(source, 
					filter_f=lambda x: x.type in ['note_on', 'note_off'], 
					unit='beat')

for msg, t in zip(msgs,times):
	print t, msg