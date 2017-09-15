from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

header = MidiFile('AutumnL.mid')

#for msg in header:
#    if msg.type != 'note_on':
#        track.append(msg.copy(time=int(msg.time*1000)))
#    else:
#        break


for i in range(1000):
    note = np.random.randint(60, 100)
    # key down
    track.append(Message('note_on', 
          note=note,
          velocity=np.random.randint(30, 70),
          time=0 if np.random.rand() > 0.8 else np.random.randint(100, 300),
          ))

    # key up 
    track.append(Message('note_on', 
          note=note,
          velocity=0,
          time=np.random.randint(100, 300),
          ))

mid.save('new_song.mid')
