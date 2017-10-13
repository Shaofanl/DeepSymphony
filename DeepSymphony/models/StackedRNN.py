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
                 block=LSTM, block_kwargs={},
                 embedding=None, embedding_w=None):
        self.input_dim = input_dim
        self.timespan = timespan
        self.output_dim = output_dim
        self.cells = cells
        self.embedding = embedding
        self.embedding_w = embedding_w
        self.block = block
        self.block_kwargs = block_kwargs

    def build(self,
              generator=False,
              **kwargs):
        if generator:
            x = input = Input(batch_shape=(1, 1) if self.embedding else
                              (1, 1, self.input_dim))
        else:
            x = input = Input((self.timespan,) if self.embedding else
                              (self.timespan, self.input_dim))

        if self.embedding:
            x = self.embedding(x)
            if self.embedding_w:
                self.embedding.set_weights(self.embedding_w)

        for cell in self.cells:
            x = self.block(cell,
                           stateful=generator,
                           return_sequences=True,
                           implementation=2,
                           **self.block_kwargs)(x)
        score = x = TimeDistributed(Dense(self.output_dim))(x)
        x = TimeDistributed(Activation('softmax'))(x)

        if generator:
            model = Model(input, score)
            self.generator = model
        else:
            model = Model(input, x)
            self.model = model
        model.summary()
        return model

    def build_generator(self,
                        weight_path=None,
                        **kwargs):
        gen = self.build(generator=True, **kwargs)
        if weight_path is None:
            gen.set_weights(self.model.get_weights())
        else:
            print 'loading weights from {}'.format(weight_path)
            gen.load_weights(weight_path)
        return gen

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

    def _generate(self,
                  gen,
                  prefix,
                  rng,
                  temperature,
                  length):
        '''
            use to avoid blocking in GTK
        '''
        if prefix:
            for note in prefix:
                res = gen.predict(np.expand_dims(note, 0))
        else:
            if self.embedding:
                res = gen.predict(rng.randint(self.input_dim, size=(1, 1)))
            else:
                res = gen.predict(np.zeros(gen.input_shape))

        iteration = 0
        while iteration < length:
            iteration += 1
            note = res[0][-1]
            note = np.exp(note/temperature)
            note /= note.sum()

            ind = rng.choice(len(note), p=note)
            note = np.zeros_like(note)
            note[ind] = 1
            if self.embedding:
                res = gen.predict(np.array([[note.argmax()]]))
            else:
                res = gen.predict(np.array([[note]]))
            yield note

    def generate(self,
                 prefix=[], seed=32,
                 temperature=1.0, length=1000,
                 max_sustain=2.0,
                 verbose=1,
                 callbacks=[], return_yield=False):
        assert hasattr(self, 'generator'),\
            "Please call build_generator() first"
        gen = self.generator

        rng = np.random.RandomState(seed)
        yielding = self._generate(gen=gen,
                                  rng=rng,
                                  prefix=prefix,
                                  length=length,
                                  temperature=temperature)
        if return_yield:
            return yielding
        notes = []
        for note in yielding:
            notes.append(note)
            if verbose:
                print '\n'.join(
                    map(lambda x: ''.join(['x' if n > 0 else '_' for n in x]),
                        [note[:128], note[128:256], note[256:356], note[356:]])
                )

            for callback in callbacks:
                if hasattr(callback, '__call__'):
                    callback(self.generator, note)
                elif hasattr(callback, '_generator_callback'):
                    callback._generator_callback(self.generator, note)

        # post process
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
