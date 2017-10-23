import numpy as np

from DeepSymphony.models.SeqAE import (
    SeqAE, SeqAEHParam)
from DeepSymphony.utils.BatchProcessing import map_dir
from DeepSymphony.utils.MidoCoder import ExampleCoder
from DeepSymphony.utils.MidoWrapper import get_midi, save_midi


if __name__ == '__main__':
    mode = 'train'
    # mode = 'eval'
    # mode = 'collect'
    mode = 'generate'

    hparam = SeqAEHParam(batch_size=64,
                         encoder_cells=[256],
                         decoder_cells=[256],
                         timesteps=1000,
                         learning_rate=2e-3,
                         iterations=1200,
                         vocab_size=363,
                         debug=False,
                         overwrite_workdir=True)
    model = SeqAE(hparam)
    model.build()
    coder = ExampleCoder()

    if mode in ['train', 'collect', 'eval']:
        data = np.array(map_dir(lambda fn: coder.encode(get_midi(fn)),
                                './datasets/easymusicnotes/'))
        print(map(lambda x: x.shape, data))

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
            model.train(fetch_data, continued=True)
        if mode == 'collect':
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
        piece_id = 1

        result = model.generate(collection[collection_id])[piece_id]
        # result = seqs[collection_id][piece_id]

        result = coder.decode(result)
        save_midi('example.mid', result)

        # how to generate?
        #   encode with 100-length seq, and decode with 1000-length seq
