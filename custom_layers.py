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

    def __init__(self, input_scale, input_zp, output_scale, output_zp, **kwargs):
        super().__init__(**kwargs)
        self.input_scale = input_scale
        self.input_zp = input_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        # input params will be same as output params (most likely)

    def call(self, inputs: tf.Tensor):
        # Logic to flatten inputs was borrowed from TF's implementation
        input_shape = inputs.shape
        non_batch_dims = input_shape[1:]
        assert non_batch_dims.is_fully_defined()
        last_dim = int(functools.reduce(operator.mul, non_batch_dims))  # 28x28=784
        flattened_shape = tf.constant([-1, last_dim])
        y = tf.reshape(inputs, flattened_shape)

        # quantize and dequantize y using self.scale and self.zero_point
        int8_val = tf.cast(tf.round(y / self.input_scale), tf.int8) + self.input_zp
        y = (tf.cast(int8_val, tf.float32) - self.output_zp) * self.output_scale
        # WIP - using next line instead of above line
        # causes drop in accuracy, even though next layer uses unquantized inputs
        # y = tf.cast((int8_val - self.output_zp), tf.float32) * self.output_scale

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
        input_scale=1,
        input_zero_point=0,
        kernel_scale=1,
        kernel_zero_point=0,
        bias_scale=1,
        bias_zero_point=0,
        output_scale=1,
        output_zero_point=0,
        **kwargs,
    ):
        super().__init__(units, activation, **kwargs)

        self.input_scale = input_scale
        self.input_zero_point = input_zero_point
        self.kernel_scale = kernel_scale
        self.kernel_zero_point = kernel_zero_point
        self.bias_scale = bias_scale
        self.bias_zero_point = bias_zero_point
        self.output_scale = output_scale
        self.output_zero_point = output_zero_point

    def call(self, inputs: tf.Tensor):

        # quantize using scale and zero_point

        # From viewing the values of the bias of first layer in netron, we know that
        # the tf.round() op is needed. (318... -> 319)
        inputs_mod = (
            tf.cast(tf.round(inputs / self.input_scale), tf.int8)
            + self.input_zero_point
        )
        kernel = (
            tf.cast(tf.round(self.kernel / self.kernel_scale), tf.int8)
            + self.kernel_zero_point
        )
        bias = (
            tf.cast(tf.round(self.bias / self.bias_scale), tf.int32)
            + self.bias_zero_point
        )

        # Dequantize - for testing. TODO: remove this section once accuracy fixes
        inputs_mod = (
            tf.cast((inputs_mod - self.input_zero_point), tf.float32) * self.input_scale
        )
        kernel = (
            tf.cast((kernel - self.kernel_zero_point), tf.float32) * self.kernel_scale
        )
        bias = tf.cast((bias - self.bias_zero_point), tf.float32) * self.bias_scale

        # Use regular matmul and addition
        y: tf.Tensor = tf.matmul(  # TODO: use inputs_mod here
            tf.cast(inputs, tf.float32), tf.cast(kernel, tf.float32)
        )
        y = tf.nn.bias_add(y, tf.cast(bias, tf.float32))
        # Outputs will have float32 type, but will be whole numbers like int32 etc
        if self.activation is not None:
            y = self.activation(y)

        # Dequantize outputs
        # y = (y - self.output_zero_point) * self.output_scale

        return y
