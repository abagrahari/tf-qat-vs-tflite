import functools
import operator

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import activations, initializers


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


def quant_from_tflite_params(
    x_fp32: tf.Tensor, scale: float, zero_point: int, dtype=tf.int8
) -> tf.Tensor:
    """Quantize a tensor, given parameters from the quantized TFLite model

    From fp32 to dtype
    """
    # From viewing the values of the bias of first layer in netron, we know that
    # the tf.round() op is needed. (e.g. 318.xx.. rounds to -> 319 rather than 318)
    return tf.cast(tf.round(x_fp32 / scale), dtype) + zero_point


def dequant_from_tflite_params(
    x_int8: tf.Tensor, scale: float, zero_point: int
) -> tf.Tensor:
    """Dequantize a tensor, given parameters from the quantized TFLite model

    From int8 to fp32
    """
    # For dequantization: cast first, then subtract zero_point after, to avoid int overflow
    return (tf.cast(x_int8, tf.float32) - zero_point) * scale


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
        y = quant_from_tflite_params(y, self.input_scale, self.input_zp, tf.int8)
        y = dequant_from_tflite_params(y, self.input_scale, self.input_zp)
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
        input_zp=0,
        kernel_scale=1,
        kernel_zp=0,
        bias_scale=1,
        bias_zp=0,
        output_scale=1,
        output_zp=0,
        **kwargs,
    ):
        super().__init__(units, activation, **kwargs)

        self.input_scale = input_scale
        self.input_zp = input_zp
        self.kernel_scale = kernel_scale
        self.kernel_zp = kernel_zp
        self.bias_scale = bias_scale
        self.bias_zp = bias_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        self.mode = "dequant_before"  # For debugging: either "dequant_before", "wrong_domain", "gary", or "mine"

    def call(self, inputs: tf.Tensor):

        # quantize using scale and zero_point
        quant_input = quant_from_tflite_params(
            inputs, self.input_scale, self.input_zp, tf.int8
        )
        quant_kernel = quant_from_tflite_params(
            self.kernel, self.kernel_scale, self.kernel_zp, tf.int8
        )
        quant_bias = quant_from_tflite_params(
            self.bias, self.bias_scale, self.bias_zp, tf.int32
        )
        if self.mode == "dequant_before":
            # Dequantize - only used for debugging
            dequant_input = dequant_from_tflite_params(
                quant_input, self.input_scale, self.input_zp
            )
            dequant_kernel = dequant_from_tflite_params(
                quant_kernel, self.kernel_scale, self.kernel_zp
            )
            dequant_bias = dequant_from_tflite_params(
                quant_bias, self.bias_scale, self.bias_zp
            )
            # Use regular matmul and addition
            y: tf.Tensor = tf.matmul(dequant_input, dequant_kernel)
            y = tf.nn.bias_add(y, dequant_bias)
            return y

        if self.mode == "wrong_domain":
            # This approach will not work, since it would be incorrectly converting between
            # input and ouput quantization domains.
            # quant_input and quant_kernel are not enough to know the quantization domains
            # We must also account for scale_i, scale_kernel, zp_i, zp_kernel to
            # convert between domains

            # Use regular matmul and addition
            y: tf.Tensor = tf.matmul(
                tf.cast(quant_input, tf.int32), tf.cast(quant_kernel, tf.int32)
            )
            y = tf.nn.bias_add(y, quant_bias)
            # Outputs will have int32 type.
            # Dequantize outputs
            y = dequant_from_tflite_params(y, self.output_scale, self.output_zp)
            return y

        if self.mode == "gary":
            # Formula per Gary's method - essentially just: dequant(inputs_int) * dequant(kernel_int) + dequant(bias_int)
            # i.e. exact same as dequantizing all the values above, before using them (mode="dequant_before")
            # It also gives the exact same output
            y = (
                tf.matmul(
                    tf.cast(quant_input, tf.float32) - self.input_zp,
                    tf.cast(quant_kernel, tf.float32) - self.kernel_zp,
                )
                * self.input_scale
                * self.kernel_scale
            ) + dequant_from_tflite_params(quant_bias, self.bias_scale, self.bias_zp)
            return y
