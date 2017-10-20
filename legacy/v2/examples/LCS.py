from DeepSymphony.utils import Song
from DeepSymphony.utils.stat import LCS
from tqdm import tqdm
from pprint import pprint
from DeepSymphony.coders import AllInOneCoder

import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    coder = AllInOneCoder()
    song = Song('masterpiece/0035.mid')
    song = coder.encode(song.midi)

    data, filelist = Song.load_from_dir(
        "./datasets/easymusicnotes/",
        encoder=AllInOneCoder(return_indices=True),
        return_list=True)

    # LCS check
    song = filter(lambda x: x < 128, song.argmax(1))
    print len(song)
    matches = []
    for ind, ele in tqdm(enumerate(data)):
        ele = filter(lambda x: x < 128, ele)
        matches.append((LCS(ele, song, return_seq=True),
                        filelist[ind]))
    matches = sorted(matches, key=lambda x: x[0][0])[::-1]
    # pprint(sorted(matches)[::-1])

    for (count, lcs, p, q), filename in matches:
        print count
        img = np.zeros((2, max(len(p), len(q))))
        img[0, :len(p)] = p*0.7+0.3
        img[1, :len(q)] = q*0.7+0.3

        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        fig.subplots_adjust(left=0., right=1., bottom=0., top=1.)
        ax.axis('off')
        ax.imshow(img, interpolation=None, aspect='auto', cmap='Reds')
        plt.show()
