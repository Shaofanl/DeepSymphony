from __future__ import print_function

import sys
from DeepSymphony.utils import Song
import mido


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.argv.append('simple_rnn.mid')
    song = Song(sys.argv[1])

    # run `timidity -iA` to open the daemon synthesizer
    # use port to send message to the daemon synthesizer

    port = mido.open_output(name='TiMidity port 0')
    keyboard = ['_']*128
    for msg in song.playback():
        if msg.type == 'note_on':
            keyboard[msg.note] = '^'
        elif msg.type == 'note_off':
            keyboard[msg.note] = '_'
        print(''.join(keyboard), end='\r')
        sys.stdout.flush()

        port.send(msg)
