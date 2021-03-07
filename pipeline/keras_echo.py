import tensorflow
import tensorflow_addons as tfa


def keras_esn(input_data):

    esn_model = tfa.layers.ESN(
            units: tfa.types.TensorLike,
            connectivity: tfa.types.FloatTensorLike = 0.1,
            leaky: tfa.types.FloatTensorLike = 1,
            spectral_radius: tfa.types.FloatTensorLike = 0.9,
            use_norm2: bool = False,
            use_bias: bool = True,
            activation: tfa.types.Activation = 'tanh',
            kernel_initializer: tfa.types.Constraint = 'glorot_uniform',
            recurrent_initializer: tfa.types.Constraint = 'glorot_uniform',
            bias_initializer: tfa.types.Constraint = 'zeros',
            return_sequences=False,
            go_backwards=False,
            unroll=False,
            **kwargs
            )

    esn_fit = esn_model.fit(input_data)

    return esn_model