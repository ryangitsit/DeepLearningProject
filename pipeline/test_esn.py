import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# inputs = np.random.random([30,23,9]).astype(np.float32)
# ESNCell = tfa.rnn.ESNCell(4)
# rnn = tf.keras.layers.RNN(ESNCell, return_sequences=True, return_state=True)


# outputs, memory_state = rnn(inputs)
# outputs.shape

# memory_state.shape