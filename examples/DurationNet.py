import numpy as np

from DeepSymphony.models.Refine import (
    RefineNet, RefineNetHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
# from DeepSymphony.utils.sample import remove_con_dup
from sklearn.model_selection import train_test_split
import music21 as ms


if __name__ == '__main__':
    hparam = RefineNetHParam(
        vocab_size=128,
        cells=[128, 64],
        output_size=16,
        timesteps=10,
        iterations=5000,
        learning_rate=1e-4,
        continued=False,
        overwrite_workdir=True,
        workdir='./temp/DurationNet/',
    )
    mode = 'train'
    mode = 'refine'
    mode = 'oneshot'

    model = RefineNet(hparam)
    model.build()

    coder = NoteDurationCoder(normalize_key='C5',
                              first_voice=False)

    try:
        data = np.load('temp/easy_with_duration.npz')
        notes = data['notes']
        durations = data['durations']
    except:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn)),
            './datasets/easymusicnotes/'))
        notes, durations = zip(*data)
        np.savez('temp/easy_with_duration.npz',
                 notes=notes,
                 durations=durations)

    notes = filter(lambda x: len(x) > hparam.timesteps, notes)
    durations = filter(lambda x: len(x) > hparam.timesteps, durations)
    notes = np.array(notes)
    durations = np.array(durations)

    train_ind, test_ind = train_test_split(np.arange(len(notes)),
                                           test_size=0.22,
                                           random_state=32)

    train_set = (notes[train_ind], durations[train_ind])
    test_set = (notes[test_ind], durations[test_ind])

    def fetch_data_g(dataset):
        def fetch_data(batch_size):
            inputs = []
            labels = []
            for _ in range(batch_size):
                ind = np.random.randint(len(dataset[0]))
                start = np.random.randint(dataset[0][ind].shape[0] -
                                          hparam.timesteps-1)
                input = dataset[0][ind][start:start+hparam.timesteps]
                label = dataset[1][ind][start:start+hparam.timesteps]
                inputs.append(input)
                labels.append(label)
            return np.array(inputs), np.array(labels)
        return fetch_data

    fetch_train_data = fetch_data_g(train_set)
    fetch_test_data = fetch_data_g(test_set)
    fetch_one_data = fetch_data_g(([train_set[0][1]],
                                   [train_set[1][1]]))

    if mode == 'train':
        model.train(fetch_one_data,
                    fetch_test_data,
                    continued=hparam.continued)
    if mode == 'refine':
        ind = train_ind[1]
        song = notes[ind]
        dura = durations[ind]
        pred_dura = model.predict([song])[0]
        print pred_dura

        coder.decode(song, 2).write('midi', 'quantized.mid')
        coder.decode(song, dura).write('midi', 'truth.mid')
        coder.decode(song, pred_dura).write('midi', 'refine.mid')
    if mode == 'oneshot':
        song = np.load('temp/oneshot_example.npz')['song']
        pred_dura = model.predict([song], max=8)[0]
        coder.decode(song, pred_dura).write('midi', 'theone.mid')
