from .BaseModel import BaseModel
from keras.layers import LSTM, TimeDistributed,\
        Dense, Activation, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.backend import get_session
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

    def reset_generator(self):
        sess = get_session()
        for layer in self.generator.layers:
            if hasattr(layer, 'states'):
                for state in layer.states:
                    zero = np.zeros(state.shape.as_list())
                    sess.run(state.assign(zero))

    def train(self,
              data_generator,
              opt=1e-4, steps_per_epoch=20, epochs=500,
              save_path='',):
        if not hasattr(self, 'model'):
            self.build()

        x, y = data_generator.next()

        if isinstance(opt, float):
            opt = Adam(opt)

        if y.shape[-1] == 1:
            self.model.compile(loss='sparse_categorical_crossentropy',
                               optimizer=opt)
        else:
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=opt)

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
                  addition,
                  gen,
                  prefix,
                  rng,
                  temperature,
                  length):
        '''
            use to avoid blocking in GTK
        '''
        if len(prefix) > 0:
            for note in prefix:
                if len(self.generator.input_shape) == 2:
                    # TODO: embedding is currently not compatible with
                    # additional bits
                    res = gen.predict(np.array([note]))
                else:
                    if addition is not None:
                        note = np.hstack([note, addition])
                    res = gen.predict(np.array([[note]]))
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

            # conditional bits
            if addition is not None:
                note = np.hstack([note, addition])

            # recursive
            if self.embedding:
                res = gen.predict(np.array([[note.argmax()]]))
            else:
                res = gen.predict(np.array([[note]]))
            yield note

    def generate(self,
                 addition=None,
                 prefix=[], seed=32,
                 temperature=1.0, length=1000,
                 verbose=1,
                 callbacks=[], return_yield=False):
        assert hasattr(self, 'generator'),\
            "Please call build_generator() first"
        gen = self.generator

        rng = np.random.RandomState(seed)
        yielding = self._generate(addition=addition,
                                  gen=gen,
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
        return np.array(notes)
