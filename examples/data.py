import numpy as np
import music21 as ms
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.sample import remove_con_dup


if __name__ == '__main__':
    coder = NoteDurationCoder(
        normalize_key='C5',
        single=False,
        first_voice=False,
    )

    try:
        data = np.load('temp/easy.npz')['data']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn))[0],
            './datasets/easymusicnotes/'))
        np.savez('temp/easy.npz', data=data)

    data = filter(len, data)

    if 0:
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

    for song in data:
        song = remove_con_dup(song)

        splits = (song == 128).nonzero()[0]
        max_margin = (splits[1:] - splits[:-1]).max()
        print max_margin
