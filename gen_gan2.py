from midiwrapper import Song
from keras.models import load_model
import numpy as np

if __name__ == '__main__':
    m = load_model('./temp/gan2.h5')
    m.summary()
    bs, seq_len, note_dim, _ = m.output_shape

    code_dim = m.input_shape[-1]

    def code_generator(bs):
        Z = np.random.uniform(-1., 1., size=(bs, code_dim))
        return Z

    mid = Song()
    track = mid.add_track()
    notes = m.predict(code_generator(1))[0, :, :, 0]

    mid._compose(track, notes, deltat=100, threshold=1.)
    mid.save_as('gan2.mid')
