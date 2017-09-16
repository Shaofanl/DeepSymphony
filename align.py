from mido import Message, MidiFile, MidiTrack, MetaMessage
from utils import getAbsT, getHots
from itertools import izip
from time import sleep

source = MidiFile('songs/bach_846.mid')
#source = MidiFile('songs/AutumnL.mid')
#source = MidiFile('datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid')
msgs, times = getAbsT(source, 
					filter_f=lambda x: x.type in ['note_on', 'note_off'], 
					unit='beat')

for msg, t in zip(msgs,times):
	print t, msg

hots = getHots(msgs, times, resolution=0.25)
print hots.shape

for hoti in hots:
	hoti = ''.join([('_' if char == 0 else 'x') for char in hoti])
	print hoti
	sleep(0.2)
