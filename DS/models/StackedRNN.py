from .BaseModel import BaseModel
from keras.layers import LSTM, TimeDistributed,\
        Dense, Activation, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np


class StackedRNN(BaseModel):
    def __init__(self,
                 timespan,
                 input_dim,
                 output_dim,
                 cells=[512, 512, 512],
                 block=LSTM):
        self.block = block
        self.input_dim = input_dim
        self.timespan = timespan
        self.output_dim = output_dim
        self.cells = cells

    def build(self, generator=False, **kwargs):
        if generator:
            x = input = Input(batch_shape=(1, 1, self.input_dim))
        else:
            x = input = Input((self.timespan, self.input_dim))

        for cell in self.cells:
            x = self.block(cell,
                           stateful=generator,
                           return_sequences=True,
                           implementation=2)(x)
        x = TimeDistributed(Dense(self.output_dim))(x)
        x = TimeDistributed(Activation('softmax'))(x)

        model = Model(input, x)
        if not generator:
            self.model = model
        return model

    def train(self,
              data_generator,
              lr=1e-4, steps_per_epoch=20, epochs=500,
              save_path='',):
        if not hasattr(self, 'model'):
            self.build()

        x, y = data_generator.next()

        if y.shape[-1] == 1:
            self.model.compile(loss='sparse_categorical_crossentropy',
                               optimizer=Adam(lr))
        else:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr))

        callbacks = []
        if save_path:
            checkpoint = ModelCheckpoint(filepath=save_path,
                                         monitor='loss',
                                         verbose=1,
                                         save_best_only=True,)
            callbacks.append(checkpoint)
        self.model.fit_generator(data_generator,
                                 steps_per_epoch=steps_per_epoch,
                                 epochs=epochs,
                                 callbacks=callbacks)

    def generate(self,
                 weight_path=None,
                 prefix=[], seed=32,
                 temperature=1.0, length=1000,
                 max_sustain=2.0):
        gen = self.build(generator=True)

        if weight_path is None:
            gen.set_weights(self.model.get_weights())
        else:
            gen.load_weights(weight_path)

        np.random.seed(seed)

        if prefix:
            for note in prefix:
                res = gen.predict(np.expand_dims(note, 0))
        else:
            res = gen.predict(np.zeros(gen.input_shape))

        notes = []
        for _ in range(length):
            note = res[0][-1]
            note = np.exp(note/temperature)
            note /= note.sum()

            ind = np.random.choice(len(note), p=note)
            note = np.zeros_like(note)
            note[ind] = 1
            res = gen.predict(np.array([[note]]))

            print '\n'.join(
                map(lambda x: ''.join(['x' if n > 0 else '_' for n in x]),
                    [note[:128], note[128:256], note[256:356], note[356:]])
            )

            notes.append(note)

        # handle
        last_appear = np.ones((128,)) * (-1)
        post_process = []
        current_t = 0.
        for note in notes:
            post_process.append(note)
            note = note.argmax()
            # print note
            if note < 128:
                if last_appear[note] == -1:
                    last_appear[note] = current_t
            elif note < 256:
                last_appear[note-128] = -1
            elif note < 356:
                current_t += (note-256)*0.1

            for key in range(128):
                if last_appear[key] > 0 and \
                   current_t - last_appear[key] > max_sustain:
                    # print('force disable {}'.format(key))
                    stop = np.zeros((363,))
                    stop[key+128] = 1.
                    last_appear[key] = -1
                    post_process.append(stop)
        return post_process
