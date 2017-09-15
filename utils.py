import mido

def copy(source, track, filter_f=lambda x: True, coef=1000):
    """
        Copy the notes from source to target track

        source: source midfile
        target: target track
        filter_list: filter for msg to be copied, copy all msg
            if it equals to None
        coef:   coefficient of time (double -> int)
    """
    for msg in source:
        if filter_f(msg):
            track.append(msg.copy(time=int(msg.time*coef)))


def getAbsT(source, filter_f=lambda x: True, unit='second'):
    """
        Translate a relative time-format into an
            absolute time-format.

        source: source MIDI object
        filter: filter of notes
        beat: turn time into beat unit 

        check http://mido.readthedocs.io/en/latest/midi_files.html?highlight=tick2second

        Minutes |                               |
        --------|-------------------------------| beats per minute (BPM=4)
        Beats   | x   x   x   x | x   x   x   x | 
        --------|-------------------------------| ticks per beat (TPB=3)
        Ticks   |^^^|^^^|^^^|^^^|^^^|^^^|^^^|^^^| or pulses per quarter note (PPQ=3)

        60000 / (BPM * PPQ)
        (i.e. a 120 BPM track would have a MIDI time of (60000 / (120 * 192)) or 2.604 ms for 1 tick.
    """

    tempo = 500000 # 120BPM

    timestamps = []
    T = 0.0
    messages = []
    for msg in source:
        if msg.type == 'set_tempo':
            tempo = msg.tempo 

        if filter_f(msg):
            if msg.time > 0: 
                if unit=='second':
                    T += msg.time 
                elif unit=='beat':
                    T += msg.time*1000000/tempo # dont know why

            messages.append(msg)
            timestamps.append(T)

    return messages, timestamps
