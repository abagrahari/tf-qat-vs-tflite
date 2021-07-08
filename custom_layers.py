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


def fake_quant(
    x: tf.Tensor,
    scale: float,
    zero_point: int,
    bits=8,
    narrow=False,
    symmetric=False,
    min_spec=-128,
) -> tf.Tensor:
    """FakeQuantize a tensor using built-in tf functions and parameters from a tflite model.

    Args:
      x: tf.Tensor to quantize
      scale: `scale` quantization parameter, from tflite
      zero_point: `zero-point` quantization parameter, from tflite
      bits: bitwidth of the quantization; between 2 and 16, inclusive
      narrow: bool; narrow_range arg of fake_quant_with_min_max_vars
      symmetric: Whether to symmetrically quantize the tensor (i.e. should min=-max?)
      min_spec: 'min' value of the range of the quantized tensor, as defined in tflite's quantization spec
    """
    # Calculate min/max from tflite params
    min = (min_spec - zero_point) * scale
    max = (127 - zero_point) * scale
    if symmetric and narrow:
        # Based on the AllValuesQuantize here https://git.io/Jc9v9
        min = tf.math.minimum(min, max * -1)
        max = tf.math.maximum(max, min * -1)
    # FakeQuantWithMinMaxVars requires that 0.0 is always in the [min; max] range.
    range_min = tf.math.minimum(min, 0.0)
    range_max = tf.math.maximum(0.0, max)
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
      input_scale: `scale` quantization parameter for the input
      input_zp: `zero-point` quantization parameter for the input
      kernel_scale: `scale` quantization parameter for the weights
      kernel_zp: `zero-point` quantization parameter for the weights
      bias_scale: `scale` quantization parameter for the bias
      bias_zp: `zero-point` quantization parameter for the bias
      output_scale: `scale` quantization parameter for the output
      output_zp: `zero-point` quantization parameter for the output

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
        kernel_tflite=None,
        bias_tflite=None,
        **kwargs,
    ):
        super().__init__(units, activation, **kwargs)

        self.input_scale = input_scale
        self.input_zp = input_zp
        self.kernel_scale = kernel_scale
        self.kernel_zp = kernel_zp
        self.bias_scale = bias_scale
        # Note: bias_scale = input_scale * kernel_scale as per tflite quantization spec
        self.bias_zp = bias_zp
        self.output_scale = output_scale
        self.output_zp = output_zp
        self.kernel_tflite = kernel_tflite
        self.bias_tflite = bias_tflite
        self.mode = "FakeQuant"
        self.debug = False  # For debugging overflow issues in layer

    def call(self, inputs: tf.Tensor):

        # Verify that kernel and bias are quantizing correctly
        quant_kernel = quant_from_tflite_params(
            self.kernel, self.kernel_scale, self.kernel_zp, tf.int8
        )
        quant_bias = quant_from_tflite_params(
            self.bias, self.bias_scale, self.bias_zp, tf.int32
        )
        # Verify quantized kernel matches int8 kernel from tflite
        tf.debugging.assert_equal(quant_kernel, self.kernel_tflite)
        # Verify quantized bias matches int32 bias from tflite
        tf.debugging.assert_equal(quant_bias, self.bias_tflite)
        # Verify that quantizing the FakeQuantized kernel matches int8 kernel from tflite
        fq_kernel = fake_quant(
            self.kernel,
            self.kernel_scale,
            self.kernel_zp,
        )
        quant_kernel = quant_from_tflite_params(
            fq_kernel, self.kernel_scale, self.kernel_zp, tf.int8
        )
        tf.debugging.assert_equal(quant_kernel, self.kernel_tflite)

        # the formula in ./tflite_formula_derivation.pdf is i.e. quant+dequant before using the values
        if self.mode == "DequantQuant":
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
            if self.activation is not None:
                y = self.activation(y)
            # quantize and dequantize the outputs to account for loss of information when quantizing
            # This is redundant on all layers except the final output layer
            # I've left it in for now to make it easier to compare this layer's intermediate outputs with the tflite model
            y = quant_from_tflite_params(y, self.output_scale, self.output_zp)
            y = dequant_from_tflite_params(y, self.output_scale, self.output_zp)
            return y
        if self.mode == "FakeQuant":
            # https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications
            fq_input = fake_quant(inputs, self.input_scale, self.input_zp)
            fq_kernel = fake_quant(
                self.kernel,
                self.kernel_scale,
                self.kernel_zp,
                # narrow=True,  # tflite spec says it uses narrow_range for weights, with below value
                # min_spec=-127,
            )
            # fake_quant_with_min_max_vars does not quantize to 32 bits
            quant_bias = quant_from_tflite_params(
                self.bias, self.bias_scale, self.bias_zp, tf.int32
            )
            dequant_bias = dequant_from_tflite_params(
                quant_bias, self.bias_scale, self.bias_zp
            )
            # Use regular matmul and addition
            y: tf.Tensor = tf.matmul(fq_input, fq_kernel)
            if self.debug:
                tf.print(y)

            y = tf.nn.bias_add(y, dequant_bias)
            if self.activation is not None:
                y = self.activation(y)
            # FakeQuant the outputs to account for loss of information when quantizing
            # This is redundant on all layers except the final output layer
            # I've left it in for now to make it easier to compare this layer's intermediate outputs with the tflite model
            y = fake_quant(y, self.output_scale, self.output_zp)
            return y
