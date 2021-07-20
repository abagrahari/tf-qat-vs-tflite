# Comparing QAT and tflite quantization parameters
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

print("tensorflow", tf.__version__)

# # Setup
# Some helper functions:
# - to convert between min/max quantization parameters, and tflite's scale/zero_point parameters.
# - to fake_quantize a tensor using `tf.quantization.fake_quant_with_min_max_vars`


def print_formatted(param: str, value: float):
    print(f"{param:35} {value:>15.6f}")


def calculate_min_max_from_tflite(
    scale: float,
    zero_point: int,
    min_spec=-128,
):
    """Calculate min/max from tflite params."""
    # Formula derived from fact that tflite quantizes
    # `real_value = (int8_value - zero_point) * scale`, and setting
    # int8_value to the range possible [minspec, 127] for int8
    # See https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications  and https://arxiv.org/pdf/1712.05877.pdf
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
    scale = (max - min) / (2 ** 8 - 1)
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


tf.random.set_seed(0)
np.random.seed(0)

# Load the MNIST dataset, and normalize it.

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Normalize the images so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# # Base model

base_model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10, use_bias=False),
        keras.layers.Dense(10, use_bias=False),
        keras.layers.Dense(10, use_bias=False),
        keras.layers.Dense(10, use_bias=False),
    ]
)
base_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
base_model.fit(train_images, train_labels, epochs=1, validation_split=0.1, verbose=1)

# # TFLite Model

# Create quantized model for TFLite from the base model
def representative_dataset():
    for data in (
        tf.data.Dataset.from_tensor_slices(train_images)
        .batch(1)
        .take(-1)  # Use all of dataset
    ):
        yield [tf.dtypes.cast(data, tf.float32)]


# Fully-integer INT8 converter settings
converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8 for Coral
converter.inference_output_type = tf.int8  # or tf.uint8 for Coral
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# We get the scale&zero_point quantization parameters from the tflite model:


tensor_details = interpreter.get_tensor_details()
tflite_params = [{}, {}]
# Flatten layer
tflite_params[0]["input_scale"] = tensor_details[0]["quantization"][0]
tflite_params[0]["input_zp"] = tensor_details[0]["quantization"][1]
tflite_params[0]["output_scale"] = tensor_details[6]["quantization"][0]
tflite_params[0]["output_zp"] = tensor_details[6]["quantization"][1]
# First Dense layer
tflite_params[1]["input_scale"] = tensor_details[6]["quantization"][0]
tflite_params[1]["input_zp"] = tensor_details[6]["quantization"][1]
tflite_params[1]["kernel_scale"] = tensor_details[2]["quantization"][0]
tflite_params[1]["kernel_zp"] = tensor_details[2]["quantization"][1]
tflite_params[1]["output_scale"] = tensor_details[7]["quantization"][0]
tflite_params[1]["output_zp"] = tensor_details[7]["quantization"][1]

# # Manual Computation
# We can manually perform the computations of the Flatten layer + the first Dense layer.
# Then, we can compare the min/max of this output to the previously extracted min/max params of the tflite model.
#
# For an input `x` and kernel `w`, I manually compute `tf.matmul(x, w)` and then compute the scale/zp of the result

# Use all the mnist train_images
kernel = base_model.weights[0]  # Get kernel from base model
# FakeQuant kernel based on params from tflite model
fq_kernel = fake_quant(
    kernel,
    tflite_params[1]["kernel_scale"],
    tflite_params[1]["kernel_zp"],
    narrow=True,  # tflite spec says it uses narrow_range for weights, with below value
    min_spec=-127,
)
outputs = []
for image in train_images:
    # Flatten image
    image = tf.cast(tf.reshape(image, [-1, 784]), tf.float32)
    assert image.shape == (1, 784)
    fq_input = fake_quant(image, tflite_params[0]["input_scale"], tflite_params[0]["input_zp"])
    y: tf.Tensor = tf.matmul(fq_input, fq_kernel)
    assert y.shape == (1, 10)
    # no bias adddition
    # linear activation function - thus, don't apply anything
    # Not fakeQuantizing outputs, in order to compare min/max params
    outputs.append(y)
outputs = np.array(outputs)


print("\nParameters from manual computation")
params = calculate_scale_zp_from_min_max(np.min(outputs), np.max(outputs))
print(f"Scale: {params[0]}, Zero-point: {params[1]}")

print("\nParameters from tflite model")
params = (tflite_params[1]["output_scale"], tflite_params[1]["output_zp"])
print(f"Scale: {params[0]}, Zero-point: {params[1]}")

# And, it appears that the tflite model parameters don't match my expected values.
