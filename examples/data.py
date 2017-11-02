import numpy as np
import music21 as ms
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
from DeepSymphony.utils.BatchProcessing import map_dir


if __name__ == '__main__':
    coder = NoteDurationCoder(
        normalize_key='C5', single=False, first_voice=True,
    )

    try:
        raise Exception
        data = np.load('temp/easy.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn))[0],
            './datasets/easymusicnotes/'))
        np.savez('temp/easy.npz', data=data)

    freq = []
    for song in data:
        if len(song) == 0:
            continue

        key = song[song != 128] % 12
        res = np.histogram(key, np.linspace(0, 12, 13))
        freq.append(res[0])
    np.set_printoptions(edgeitems=15)
    freq = np.array(freq)
    print freq
    print freq.sum()
