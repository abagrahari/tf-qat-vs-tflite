# Running a single image through a pre-built tflite model
# Then verifying the calculations manually
# in order to gain an understanding of how quant works in tflite
# The aim is to mimic keras' dense layer

# TODO: this was left in an unfinished state, as it was no longer needed
# The quantization/dequantization and output calculation steps / formula
# may be incorrect.

import os
from pathlib import Path

from tensorflow.python.util.nest import flatten

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

import tflite_runner
import utils

SEED = 0

tf.random.set_seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

# Ensure tflite model loaded correctly
assert Path("saved_models/base_model_dense4.tflite").exists()
tflite_model = tflite_runner.create_tflite_model(
    None, None, "saved_models/base_model_dense4.tflite"
)
tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
    tflite_model, test_images, test_labels
)
assert tflite_model_accuracy == 0.9028

# Run tflite interpreter on a single image
img = train_images[1]
interpreter = tflite_runner.get_interpreter(tflite_model)
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]
# Check if the input type is quantized, then rescale input data to uint8
# as shown in TF's Post-Training Integer Quantization Example
assert input_details["dtype"] in [np.uint8, np.int8]
input_scale, input_zero_point = input_details["quantization"]
img = img / input_scale + input_zero_point
img = np.round(img)
# Pre-processing: add batch dimension and convert to datatype to match with
# the model's input data format (int8)
img = np.expand_dims(img, axis=0).astype(input_details["dtype"])
# Run inference.
interpreter.set_tensor(input_details["index"], img)
interpreter.invoke()
# Post-processing: remove batch dimension and dequantize the output
# based on TensorFlow's quantization params
# We dequantize the outputs so we can directly compare the raw
# outputs with the QAT model
output = interpreter.get_tensor(output_details["index"])[0]
if output_details["dtype"] in [np.uint8, np.int8]:
    output_scale, output_zero_point = output_details["quantization"]
    output = (output.astype(np.float32) - output_zero_point) * output_scale
output = np.array(output)


# MANUAL CALCULATION OF FIRST LAYER
tensor_details = interpreter.get_tensor_details()
input_scale = (tensor_details[10]["quantization"][0],)
input_zero_point = (tensor_details[10]["quantization"][1],)
kernel_scale = (tensor_details[2]["quantization"][0],)
kernel_zero_point = (tensor_details[2]["quantization"][1],)
bias_scale = (tensor_details[3]["quantization"][0],)
bias_zero_point = (tensor_details[3]["quantization"][1],)
output_scale = (tensor_details[11]["quantization"][0],)
output_zero_point = (tensor_details[11]["quantization"][1],)

model_input = train_images[1]
## Layer 1 Input
input_int8_tflite = interpreter.get_tensor(10)


def flatten_with_quant(scale, zero_point, input_shape, inputs: tf.Tensor):
    import functools
    import operator

    # Logic to flatten inputs was borrowed from TF's implementation
    non_batch_dims = input_shape
    last_dim = int(functools.reduce(operator.mul, non_batch_dims))  # 28x28=784
    flattened_shape = tf.constant([-1, last_dim])
    y = tf.reshape(inputs, flattened_shape)
    # quantize and dequantize y using self.scale and self.zero_point
    int8_val = tf.cast(tf.round(y / scale), tf.int8) + zero_point
    y = (tf.cast(int8_val, tf.float32) - zero_point) * scale
    return y


flatten_layer_output = flatten_with_quant(
    scale=tensor_details[0]["quantization"][0],
    zero_point=tensor_details[0]["quantization"][1],
    input_shape=(28, 28),
    inputs=model_input,
)
input_fp32_manual = flatten_layer_output
input_int8_manual = (
    np.round(input_fp32_manual / input_scale).astype(int) + input_zero_point
)
print("TFLite", input_int8_tflite)
print("Manual", input_int8_manual)
assert np.array_equal(input_int8_tflite, input_int8_manual)

# Get int weights directly from tflite model
weights_int8_tflite = interpreter.get_tensor(10).transpose()
bias_int32_tflite = interpreter.get_tensor(3)

output_int8_tflite = interpreter.get_tensor(11)


# quantize using scale and zero_point
# From viewing the values of the bias of first layer in netron, we know that
# the tf.round() op is needed. (318... -> 319)
inputs_mod = tf.cast(tf.round(img / input_scale), tf.int8) + input_zero_point

# Dequantize - for testing. TODO: remove this section once accuracy fixes
inputs_mod = tf.cast((inputs_mod - input_zero_point), tf.float32) * input_scale

# Use regular matmul and addition
y: tf.Tensor = tf.matmul(
    tf.cast(img, tf.float32), tf.cast(weights_int8_tflite, tf.float32)
)
y = tf.nn.bias_add(y, tf.cast(bias_int32_tflite, tf.float32))
# Outputs will have float32 type, but will be whole numbers like int32 etc

# Dequantize outputs
# y = (y - self.output_zero_point) * self.output_scale

output_int8_manual = y


print(
    "\nLayer 1 TfLite calculted activation\n",
    output_int8_tflite,
)

print(
    "\nLayer 1 Manually calculted activation\n",
    output_int8_manual,
)

assert np.array_equal(output_int8_tflite, output_int8_manual)


# Run test dataset on custom, and TFLite models
tflite_output = tflite_runner.run_tflite_model(tflite_model, test_images)
