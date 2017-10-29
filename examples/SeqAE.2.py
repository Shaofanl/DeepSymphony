import numpy as np

from DeepSymphony.models.SeqAE import (
    SeqAE, SeqAEHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.MidoWrapper import save_midi
from DeepSymphony.utils.Music21Coder import NoteDurationCoder
from DeepSymphony.eval.LCS import eval_lcs
import music21 as ms


if __name__ == '__main__':
    # usage:
    # 1. train a model
    # 2. evaluate it if you want
    # 3. collect the codes
    # 4. generate with the collected code
    mode = 'train'
    mode = 'eval'
    mode = 'collect'
    mode = 'generate'

    hparam = SeqAEHParam(batch_size=64,
                         encoder_cells=[256],
                         decoder_cells=[256],
                         timesteps=200,
                         gen_timesteps=1000,
                         learning_rate=5e-3,
                         iterations=2000,
                         vocab_size=128+1,
                         debug=False,
                         overwrite_workdir=True)
    model = SeqAE(hparam)
    model.build()
    # coder = ExampleCoder()
    coder = NoteDurationCoder()

    if mode in ['train', 'collect', 'eval']:
        data = np.array(map_dir(
            lambda fn: coder.encode(ms.converter.parse(fn))[0],
            './datasets/easymusicnotes/'))

        print(len(data), map(lambda x: len(x), data))
        data = filter(lambda x: len(x) > hparam.timesteps, data)
        print(len(data), map(lambda x: len(x), data))

        def fetch_data(batch_size):
            seqs = []
            for _ in range(batch_size):
                ind = np.random.randint(len(data))
                start = np.random.randint(data[ind].shape[0] -
                                          hparam.timesteps-1)
                seq = data[ind][start:start+hparam.timesteps]
                seqs.append(seq)
            return np.array(seqs)

        if mode == 'train':
            model.train(fetch_data)  # continued=True)
        if mode == 'collect':
            np.random.seed(32)
            collection, seqs = model.collect(fetch_data, samples=10)
            np.savez(hparam.workdir+'code_collection.npz',
                     wrapper={'code': collection, 'seqs': seqs})
        if mode == 'eval':
            seqs = fetch_data(hparam.batch_size)
            pred, train_pred = model.eval(seqs)
            np.set_printoptions(linewidth=np.inf)
            for i in range(hparam.batch_size):
                print '='*200
                print seqs[i]
                print train_pred[i]
                print pred[i]

    elif mode == 'generate':
        collection = np.load(hparam.workdir+'code_collection.npz').\
                __getitem__('wrapper').flatten()[0].get('code')
        seqs = np.load(hparam.workdir+'code_collection.npz').\
            __getitem__('wrapper').flatten()[0].get('seqs')

        collection_id = 0
        piece_id = 4

        result = model.generate(collection[collection_id])[piece_id]
        coder.decode(result, [2]*len(result)).write('midi',
                                                    'example.mid')

        truth = seqs[collection_id][piece_id]
        coder.decode(truth, [2]*len(truth)).write('midi',
                                                  'truth.mid')

        # find the whole truth song:
        if False:
            data = np.array(map_dir(
                lambda fn: coder.encode(ms.converter.parse(fn))[0],
                './datasets/easymusicnotes/'))
            data = filter(lambda x: len(x) > hparam.timesteps, data)

            print eval_lcs(result, data)

            def fetch_data(batch_size):
                seqs = []
                for _ in range(batch_size):
                    ind = np.random.randint(len(data))
                    start = np.random.randint(data[ind].shape[0] -
                                              hparam.timesteps-1)
                    seq = data[ind]
                    seqs.append(seq)
                return np.array(seqs)
            np.random.seed(32)
            whole = fetch_data(hparam.batch_size)[piece_id]
            coder.decode(whole, [2]*len(whole)).write('midi',
                                                      'whole.mid')

        # how to generate?
        #   encode with 100-length seq, and decode with 1000-length seq
