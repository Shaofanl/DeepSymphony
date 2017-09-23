import mido
import numpy as np
from mido import Message, MidiFile, MidiTrack, MetaMessage
from midiwrapper import Song
from time import sleep
import sys


def demo_copy(argv):
    mid = Song()
    track = mid.add_track()

    source = MidiFile('playground/AutumnL.mid')
    # only copy notes
    Song._copy(source, track, filter_f=lambda x: x.type == 'note_on')
    mid.save('playground/AutumnL_copy.mid')


def demo_align(argv):
    source = MidiFile('songs/bach_846.mid')
    # source = MidiFile('songs/AutumnL.mid')
    # source = MidiFile('datasets/easymusicnotes/level6/anniversary-song-glen-miller-waltz-piano-level-6.mid')
    msgs, times = Song._get_absolute_time(
        source,
        filter_f=lambda x: x.type in ['note_on', 'note_off'],
        unit='beat')

    for msg, t in zip(msgs, times):
        print t, msg

    hots = Song._get_hots(msgs, times, resolution=0.25)
    print hots.shape

    for hoti in hots:
        hoti = ''.join([('_' if char == 0 else 'x') for char in hoti])
        print hoti
        sleep(0.2)


def demo_playback(argv):
    if argv:
        mid = mido.MidiFile(argv[0])
    else:
        mid = mido.MidiFile('AutumnL.mid')
    for msg in mid:
        print msg


def demo_random_new(argv):
    mid = Song()
    track = mid.add_track()

    # header = MidiFile('AutumnL.mid')

    for i in range(1000):
        note = np.random.randint(60, 100)
        # key down
        track.append(
            Message('note_on',
                    note=note,
                    velocity=np.random.randint(30, 70),
                    time=0 if np.random.rand() > 0.8 else
                    np.random.randint(100, 300),)
        )

        # key up
        track.append(Message('note_on',
                             note=note,
                             velocity=0,
                             time=np.random.randint(100, 300),))
    mid.save('new_song.mid')


def demo_connect(argv):
    inport = mido.open_input()

    while 1:
        msg = inport.receive()
        if msg.type != 'clock':
            print msg


demos = [demo_copy, demo_align, demo_playback, demo_random_new, demo_connect]
names = map(lambda x: x.__name__, demos)
if __name__ == '__main__':
    argv = sys.argv
    if len(argv) < 2:
        print 'Usage: python demos.py {}'.format(names)
    else:
        argv[1] = 'demo_'+argv[1]
        if argv[1] not in names:
            raise NotImplementedError("Demo `{}` not exist.".format(argv[1]))
        else:
            f = demos[names.index(argv[1])](argv[2:])
