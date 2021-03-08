import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

from esn_cell import ESNCell


def keras_esn(inputs, outputs):

        # model = keras.Sequential()
        # # Add an Embedding layer expecting input vocab of size 1000, and
        # # output embedding dimension of size 64.
        # model.add(layers.Embedding(input_dim=1000, output_dim=64))

        # # Add a ESN layer with 128 internal units.
        # model.add(tfa.layers.ESN(128))

        # # Add a Dense layer with 10 units.
        # model.add(layers.Dense(10))

        # model.summary()

        # # how to optimize if output is just linear regression?
        # model.compile(
        # optimizer=None,
        # loss='binary_crossentropy',
        # metrics=['accuracy'],
        # )

        # run = model.fit(inputs=inputs, outputs=outputs)  # can here also specifiy epochs
        

        return None