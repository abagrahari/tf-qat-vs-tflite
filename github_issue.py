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
    # See https://www.tensorflow.org/lite/performance/quantization_spec#int8_quantized_operator_specifications and https://arxiv.org/pdf/1712.05877.pdf
    min = (min_spec - zero_point) * scale
    max = (127 - zero_point) * scale
    # FakeQuantWithMinMaxVars requires that 0.0 is always in the [min; max] range.
    # See https://git.io/JWKjb
    range_min = tf.math.minimum(min, 0.0)
    range_max = tf.math.maximum(0.0, max)
    return range_min, range_max


def calculate_scale_zp_from_min_max(min, max):
    """Calculate scale and zero-point from asymmetric min/max.
    Note: will not work for parameters created with narrow_range.
    """
    quant_min = -128  # std::numeric_limits<int8_t>::min()
    quant_max = 127  # std::numeric_limits<int8_t>::max()
    # scale = (max - min) / (2 ** 8 - 1) # formula from Section 3 in https://arxiv.org/pdf/1712.05877.pdf

    # Below is borrowed from TfLite's GetAsymmetricQuantizationParams https://git.io/JBcVy
    # Adjust the boundaries to guarantee 0 is included.
    min = tf.math.minimum(min, 0)
    max = tf.math.maximum(max, 0)
    scale = (max - min) / (quant_max - quant_min)
    zero_point_from_min = quant_min
    if scale != 0:
        zero_point_from_min = quant_min - min / scale
    if zero_point_from_min < quant_min:
        zero_point = quant_min
    elif zero_point_from_min > quant_max:
        zero_point = quant_max
    else:
        zero_point = np.round(zero_point_from_min)
    return scale, int(zero_point)


def calculate_nudged_params(min, max, narrow_range=False):
    """Calculate nudged min,max, and scale from asymmetric min/max."""
    # Below is borrowed from TF's FakeQuantWithMinMaxArgs https://git.io/JBCs4, https://git.io/JBCiI, https://git.io/JBCsQ
    quant_min = 1 if narrow_range else 0
    quant_max = (2 ** 8) - 1  # 255

    # Nudge()
    scale = (max - min) / (quant_max - quant_min)
    zero_point_from_min = quant_min - min / scale
    if zero_point_from_min < quant_min:
        nudged_zero_point = quant_min
    elif zero_point_from_min > quant_max:
        nudged_zero_point = quant_max
    else:
        nudged_zero_point = tf.math.round(zero_point_from_min)
    nudged_zero_point = int(
        nudged_zero_point
    )  # will not match zp from GetAsymmetricQuantizationParams b/c of quant_min and quant_max values
    nudged_min = (quant_min - nudged_zero_point) * scale
    nudged_max = (quant_max - nudged_zero_point) * scale
    # end Nudge()

    return nudged_min, nudged_max, scale, nudged_zero_point


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
# As per TfLite's QuantizeModel https://git.io/J4hxt, it seems that a full fp32 forward pass is done first
# after which, quantization parameters are independantly calculated.
outputs = []
for image in train_images:
    # Flatten image
    image = tf.cast(tf.reshape(image, [-1, 784]), tf.float32)
    assert image.shape == (1, 784)
    y: tf.Tensor = tf.matmul(image, kernel)
    assert y.shape == (1, 10)
    # no bias adddition
    # linear activation function - thus, don't apply anything
    outputs.append(y)
outputs = np.array(outputs)


print("\nParameters from manual computation")
params = calculate_scale_zp_from_min_max(np.min(outputs), np.max(outputs))
print(f"Scale: {params[0]}, Zero-point: {params[1]}")

print("\nParameters from tflite model")
params = (tflite_params[1]["output_scale"], tflite_params[1]["output_zp"])
print(f"Scale: {params[0]}, Zero-point: {params[1]}")

# ---
# Let's look at the `max/min` parameters instead.
#
# For TfLite - we will compute the `min/max` from the `scale/zp` params.
#
# For the manual computation - we will look at the `min/max` of the outputs.
# We will also convert this `min/max` to `scale/zp`, and then convert back to `min/max`. This is to
# account for the loss of info when converting from `min/max` to `scale/zp` since `zp` is an `int8`

print("\nParameters from manual computation")
params = (np.min(outputs), np.max(outputs))
print(f"True Min: {params[0]}, True Max: {params[1]}")

params = calculate_min_max_from_tflite(*calculate_scale_zp_from_min_max(*params))
print(f"Adjusted Min: {params[0]}, Adjusted Max: {params[1]}")

params = calculate_nudged_params(np.min(outputs), np.max(outputs))
print(f"Nudged Min: {params[0]}, Nudged Max: {params[1]}, Scale: {params[2]}")


print("\nParameters from tflite model")
params = (tflite_params[1]["output_scale"], tflite_params[1]["output_zp"])
params = calculate_min_max_from_tflite(*params)
print(f"Min: {params[0]}, Max: {params[1]}")

# While the true min/max don't match with tflite, it looks like the 'adjusted' and 'nudged' versions do.
