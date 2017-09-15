import mido
import sys

if len(sys.argv) > 1:
    mid = mido.MidiFile(sys.argv[1])
else:
    mid = mido.MidiFile('AutumnL.mid')

for msg in mid:
    print msg
