from .BaseModel import BaseModel
from keras.layers import Embedding,\
        Activation, Input, Lambda
from keras.layers.merge import Multiply
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K


class EmbeddingModel(BaseModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 seq_len,):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len

    def build(self, generator=False, **kwargs):
        if generator:
            input = Input((1,), dtype='int32')
            embedded = Embedding(self.input_dim,
                                 self.output_dim,
                                 input_length=1)(input)
            embed_model = Model(input, embedded)
            return embed_model
        else:
            input = Input((self.seq_len,), dtype='int32')
            embedded = Embedding(self.input_dim,
                                 self.output_dim,
                                 input_length=self.seq_len)(input)
            embed_model = Model(input, embedded)

        input1 = Input((self.seq_len,), dtype='int32')
        input2 = Input((self.seq_len,), dtype='int32')
        emb1 = embed_model(input1)  # (bs, seq_len, output_dim)
        emb2 = embed_model(input2)  # (bs, seq_len, output_dim)
        sim = Multiply()([emb1, emb2])
        sim = Lambda(lambda x: K.sum(x, 2),
                     output_shape=lambda x: x[:-1])(sim)
        sim = Activation('sigmoid')(sim)
        self.siamese = Model([input1, input2], sim)
        self.embed_model = embed_model
        self.siamese.summary()

    def train(self,
              data_generator,
              lr=1e-4, steps_per_epoch=20, epochs=500,
              save_path='',):
        if not hasattr(self, 'model'):
            self.build()

        self.siamese.compile(loss='binary_crossentropy',
                             optimizer=Adam(lr))

        callbacks = []
        if save_path:
            checkpoint = ModelCheckpoint(filepath=save_path,
                                         monitor='loss',
                                         verbose=1,
                                         save_best_only=True,)
            callbacks.append(checkpoint)
        self.siamese.fit_generator(data_generator,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=epochs,
                                   callbacks=callbacks)

    def build_generator(self, weight_path=None):
        gen = self.build(generator=True)
        if weight_path is None:
            gen.set_weights(self.siamese.get_weights())
        else:
            gen.load_weights(weight_path)
        self.generator = gen
        return gen

    def generate(self, inputs,):
        return self.generator.predict(inputs)
