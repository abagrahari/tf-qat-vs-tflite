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
      use_bias: Boolean, whether the layer uses a bias vector.

    Input shape:
      2D tensor with shape `(batch_size, input_dim)`,
    Output shape:
      2D tensor with shape: `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        use_bias=True,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

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
        if self.use_bias:
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
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        # Perform layer's computation, in the forward pass.
        # Back prop is automatically handled by tf
        # Can only use tensorflow functions here

        assert inputs.shape.rank in (2, None)
        y = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            y = tf.nn.bias_add(y, self.bias)

        if self.activation is not None:
            y = self.activation(y)
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


def dequant_from_tflite_params(x_int8: tf.Tensor, scale: float, zero_point: int) -> tf.Tensor:
    """Dequantize a tensor, given parameters from the quantized TFLite model

    From int8 to fp32
    """
    # For dequantization: cast first, then subtract zero_point after, to avoid int overflow
    return (tf.cast(x_int8, tf.float32) - zero_point) * scale


def calculate_min_max_from_tflite(
    scale: float,
    zero_point: int,
    min_spec=-128,
):
    """Calculate min/max from tflite params."""
    # Formula derived from fact that tflite quantizes
    # `real_value = (int8_value - zero_point) * scale`, and setting
    # int8_value to the range possible [minspec, 127] for int8
    min = (min_spec - zero_point) * scale
    max = (127 - zero_point) * scale
    # FakeQuantWithMinMaxVars requires that 0.0 is always in the [min; max] range.
    # See https://git.io/JWKjb
    range_min = tf.math.minimum(min, 0.0)
    range_max = tf.math.maximum(0.0, max)
    return range_min, range_max


def calculate_scale_zp_from_min_max(min, max):
    """Calculate scale and zero-point from min/max.

    Note: will not work for parameters created with narrow_range.
    """
    # Below formula is from Section 3 in https://arxiv.org/pdf/1712.05877.pdf
    scale = (max.numpy() - min.numpy()) / (2 ** 8 - 1)
    # Below formula is rearrangment of calculate_min_max_from_tflite
    zero_point = 127 - max / scale
    return scale, zero_point


def fake_quant(
    x: tf.Tensor,
    scale: float,
    zero_point: int,
    bits=8,
    narrow=False,
    min_spec=-128,
) -> tf.Tensor:
    """FakeQuantize a tensor using built-in tf functions and parameters from a tflite model.

    Args:
      x: tf.Tensor to quantize
      scale: `scale` quantization parameter, from tflite
      zero_point: `zero-point` quantization parameter, from tflite
      bits: bitwidth of the quantization; between 2 and 16, inclusive
      narrow: bool; narrow_range arg of fake_quant_with_min_max_vars
      min_spec: 'min' value of the range of the quantized tensor, as defined in tflite's quantization spec
    """
    range_min, range_max = calculate_min_max_from_tflite(scale, zero_point, min_spec)
    return tf.quantization.fake_quant_with_min_max_vars(
        x, range_min, range_max, num_bits=bits, narrow_range=narrow
    )


class FlattenTFLite(keras.layers.Layer):
    """Flattens the input.

    Needed for adding input quantization params from tflite

    Args:
      input_scale: `scale` quantization parameter for the input
      input_zp: `zero-point` quantization parameter for the input
      kernel_scale: `scale` quantization parameter for the weights
      kernel_zp: `zero-point` quantization parameter for the weights
      bias_scale: `scale` quantization parameter for the bias
      bias_zp: `zero-point` quantization parameter for the bias
      output_scale: `scale` quantization parameter for the output
      output_zp: `zero-point` quantization parameter for the output
    """

    def __init__(self, input_scale, input_zp, output_scale, output_zp, **kwargs):
        super().__init__(**kwargs)
        self.input_scale = input_scale
        self.input_zp = input_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        # TFLite quantization spec requires input params are same as output params.

    def call(self, inputs: tf.Tensor):
        # Logic to flatten inputs was borrowed from TF's implementation
        input_shape = inputs.shape
        non_batch_dims = input_shape[1:]
        assert non_batch_dims.is_fully_defined()
        last_dim = int(functools.reduce(operator.mul, non_batch_dims))  # 28x28=784
        flattened_shape = tf.constant([-1, last_dim])
        y = tf.reshape(inputs, flattened_shape)

        # quantize and dequantize y using self.scale and self.zero_point
        # y = quant_from_tflite_params(y, self.input_scale, self.input_zp, tf.int8)
        # y = dequant_from_tflite_params(y, self.input_scale, self.input_zp)
        # Seems fake_quant is identical to using above dequant(quant(...))
        y = fake_quant(y, self.input_scale, self.input_zp)

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
      use_bias: Boolean, whether the layer uses a bias vector.
      input_scale: `scale` quantization parameter for the input
      input_zp: `zero-point` quantization parameter for the input
      input_min: `min` quantization parameter for the input.
      input_max: `max` quantization parameter for the input.
      kernel_scale: `scale` quantization parameter for the weights
      kernel_zp: `zero-point` quantization parameter for the weights
      bias_scale: `scale` quantization parameter for the bias
      bias_zp: `zero-point` quantization parameter for the bias
      output_scale: `scale` quantization parameter for the output
      output_zp: `zero-point` quantization parameter for the output
      output_min: `min` quantization parameter for the output.
      output_max: `max` quantization parameter for the output.
      fq_bias: Boolean, whether to fake quantize the bias

    Input shape:
      2D tensor with shape `(batch_size, input_dim)`,
    Output shape:
      2D tensor with shape: `(batch_size, units)`.
    """

    def __init__(
        self,
        units: int,
        activation=None,
        use_bias=True,
        input_scale=None,
        input_zp=None,
        input_min=None,
        input_max=None,
        kernel_scale=None,
        kernel_zp=None,
        bias_scale=None,
        bias_zp=None,
        output_scale=None,
        output_zp=None,
        kernel_tflite=None,
        bias_tflite=None,
        output_min=None,
        output_max=None,
        fq_bias=True,
        **kwargs,
    ):
        super().__init__(units, activation, use_bias, **kwargs)

        self.input_scale = input_scale
        self.input_zp = input_zp
        self.input_min = input_min
        self.input_max = input_max
        self.kernel_scale = kernel_scale
        self.kernel_zp = kernel_zp
        self.bias_scale = bias_scale
        # Note: bias_scale = input_scale * kernel_scale as per tflite quantization spec
        self.bias_zp = bias_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        self.kernel_tflite = kernel_tflite
        self.bias_tflite = bias_tflite
        self.output_min = output_min
        self.output_max = output_max
        self.fq_bias = fq_bias
        self.print_int8_layer_outputs = False  # For debugging issues in quantization of layer

    def call(self, inputs: tf.Tensor):

        # Verify that kernel and bias are quantizing correctly
        if self.kernel_tflite is not None:
            # Verify quantized kernel matches int8 kernel from tflite
            quant_kernel = quant_from_tflite_params(
                self.kernel, self.kernel_scale, self.kernel_zp, tf.int8
            )
            tf.debugging.assert_equal(quant_kernel, self.kernel_tflite)
            # Also verify that quantizing the FakeQuantized kernel matches int8 kernel from tflite
            fq_kernel = fake_quant(
                self.kernel, self.kernel_scale, self.kernel_zp, narrow=True, min_spec=-127
            )
            quant_kernel = quant_from_tflite_params(
                fq_kernel, self.kernel_scale, self.kernel_zp, tf.int8
            )
            tf.debugging.assert_equal(quant_kernel, self.kernel_tflite)

        if self.use_bias and self.bias_tflite is not None:
            # Verify quantized bias matches int32 bias from tflite
            quant_bias = quant_from_tflite_params(
                self.bias, self.bias_scale, self.bias_zp, tf.int32
            )
            tf.debugging.assert_equal(quant_bias, self.bias_tflite)

        if self.input_min is not None and self.input_max is not None:
            # Use min/max parameters
            fq_input = tf.quantization.fake_quant_with_min_max_vars(
                inputs,
                self.input_min,
                self.input_max,
                num_bits=8,
                narrow_range=False,
            )
        else:
            # use scale/zp parameters
            fq_input = fake_quant(inputs, self.input_scale, self.input_zp)
        # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
        fq_kernel = fake_quant(
            self.kernel,
            self.kernel_scale,
            self.kernel_zp,
            narrow=True,  # tflite spec says it uses narrow_range for weights, with below value
            min_spec=-127,
        )
        y: tf.Tensor = tf.matmul(fq_input, fq_kernel)

        if self.use_bias:
            if self.fq_bias:
                # fake_quant_with_min_max_vars does not quantize to 32 bits
                # https://git.io/JW6eK
                quant_bias = quant_from_tflite_params(
                    self.bias, self.bias_scale, self.bias_zp, tf.int32
                )
                dequant_bias = dequant_from_tflite_params(
                    quant_bias, self.bias_scale, self.bias_zp
                )
                y = tf.nn.bias_add(y, dequant_bias)
            else:
                y = tf.nn.bias_add(y, self.bias)
        if self.activation is not None:
            y = self.activation(y)

        # FakeQuant the outputs to account for loss of information when quantizing
        # This is redundant on all layers except the final output layer
        # I've left it in for now to make it easier to compare this layer's intermediate outputs with the tflite model

        if self.output_min is not None and self.output_max is not None:
            # Use min/max parameters
            y = tf.quantization.fake_quant_with_min_max_vars(
                y, self.output_min, self.output_max, num_bits=8, narrow_range=False
            )
        else:
            # use scale/zp parameters
            y = fake_quant(y, self.output_scale, self.output_zp)

        if self.print_int8_layer_outputs:
            layer_num = (int(self.name[-1]) + 1) if self.name[-1] != "e" else 1
            tf.print(
                f"Custom Dense Layer {layer_num}'s int8 output:",
                quant_from_tflite_params(y, self.output_scale, self.output_zp, tf.int8),
                summarize=-1,
            )
        return y
