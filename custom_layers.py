import functools
import operator

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers


class Dense(keras.layers.Layer):
    """Regular densely-connected NN layer. Simplified from TensorFlow implementation.

    Implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).

    Input shape:
      2D tensor with shape `(batch_size, input_dim)`,
    Output shape:
      2D tensor with shape: `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        # Lazily create and init weights and biases, since shape of input tensor is now known
        # inputs.shape is (batchsize, features)
        # kernel.shape is (features, units)
        # bias.shape is (units)
        assert input_shape[-1] is not None

        # weight matrix
        self.kernel = self.add_weight(
            "kernel",  # need to specify name to be able to save/load models
            shape=[input_shape[-1], self.units],
            initializer=initializers.get("glorot_uniform"),
            dtype=self.dtype,
            trainable=True,
        )
        # bias vector
        self.bias = self.add_weight(
            "bias",
            shape=[
                self.units,
            ],
            initializer=initializers.get("zeros"),  # for bias vector
            dtype=self.dtype,
            trainable=True,
            # vector of size (units,)
        )

    def call(self, inputs):
        # Perform layer's computation, in the forward pass.
        # Back prop is automatically handled by tf
        # Can only use tensorflow functions here

        assert inputs.shape.rank in (2, None)
        y = tf.matmul(inputs, self.kernel)

        y = tf.nn.bias_add(y, self.bias)

        if self.activation is not None:
            y = self.activation(y)
        return y


class DenseFakeQuant(Dense):
    """Densely-connected NN layer, with Fake Quantization.

    Implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).

    Input shape:
      2D tensor with shape `(batch_size, input_dim)`,
    Output shape:
      2D tensor with shape: `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        **kwargs,
    ):
        super().__init__(units, activation, **kwargs)

    def quant_and_dequant(self, x: tf.Tensor, bits=8) -> tf.Tensor:
        # unused_val = 0
        # return tf.quantization.quantize_and_dequantize_v2(
        #     x, unused_val, unused_val, num_bits=bits, range_given=False
        # )

        # Find min/max of inputs
        batch_min = tf.math.reduce_min(x, name="BatchMin")
        batch_max = tf.math.reduce_max(x, name="BatchMax")
        # FakeQuantWithMinMaxVars requires that 0.0 is always in the [min; max] range.
        range_min = tf.math.minimum(batch_min, 0.0)
        range_max = tf.math.maximum(0.0, batch_max)

        return tf.quantization.fake_quant_with_min_max_vars(x, range_min, range_max)

    def call(self, inputs: tf.Tensor):
        assert inputs.shape.rank in (2, None)

        # FakeQuant inputs, kernel, and bias
        fq_inputs = self.quant_and_dequant(inputs)
        fq_kernel = self.quant_and_dequant(self.kernel)
        # fq bias to 32 bits as tflite does
        fq_bias = self.quant_and_dequant(self.bias, 32)

        # Use regular matmul
        y = tf.matmul(fq_inputs, fq_kernel)
        y = self.quant_and_dequant(y)

        # Use regular addition
        y = tf.nn.bias_add(y, fq_bias)
        y = self.quant_and_dequant(y)

        if self.activation is not None:
            y = self.activation(y)
        y = self.quant_and_dequant(y)
        return y


class FlattenTFLite(keras.layers.Layer):
    """Flattens the input.

    Needed for adding input quantization params from tflite
    """

    def __init__(self, data_format=None, **kwargs):
        super().__init__(**kwargs)
        # Hardcoded param from tflite for now
        self.scale = 0.003921568859368563
        self.zero_point = -128

    def call(self, inputs):
        # Borrowed from TF's implementation
        input_shape = inputs.shape
        non_batch_dims = input_shape[1:]
        assert non_batch_dims.is_fully_defined()
        last_dim = int(functools.reduce(operator.mul, non_batch_dims))  # 28x28=784
        flattened_shape = tf.constant([-1, last_dim])
        y = tf.reshape(inputs, flattened_shape)
        # quantize (and dequantize?) y using self.scale and self.zero_point
        tf.print(y, summarize=-1)
        int8_val = tf.math.add(tf.math.divide(y, self.scale), self.zero_point)
        int8_val = tf.round(int8_val)
        int8_val = tf.cast(int8_val, tf.int8)
        y = tf.math.scalar_mul(
            self.scale, tf.math.subtract(tf.cast(int8_val, tf.float32), self.zero_point)
        )
        return y


class DenseTFLite(Dense):
    """Densely-connected NN layer, with Quantization using params from tflite.

    Implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer.

    Args:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).

    Input shape:
      2D tensor with shape `(batch_size, input_dim)`,
    Output shape:
      2D tensor with shape: `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        **kwargs,
    ):
        super().__init__(units, activation, **kwargs)
