# Running a single image through a pre-built tflite model
# to check if tensors can be properly extracted from tflite

import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

import tflite_runner
import utils

SEED = 0

tf.random.set_seed(SEED)

saved_path = "saved_models/base_model_dense_0.tflite"
assert Path(saved_path).exists()


# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()


tflite_model = tflite_runner.create_tflite_model(None, None, saved_path)

# Run tflite interpreter on a single image
img = test_images[9070]


interpreter = tf.lite.Interpreter(
    model_content=tflite_model, experimental_preserve_all_tensors=True
)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()[0]
# Check if the input type is quantized, then rescale input data to uint8
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

tf.print("Dense layer 1 output", interpreter.get_tensor(11))
tf.print("Dense layer 2 output", interpreter.get_tensor(12))
tf.print("Dense layer 3 output", interpreter.get_tensor(13))
tf.print("Dense layer 4 output", interpreter.get_tensor(14))
assert not np.array_equal(
    interpreter.get_tensor(12).flatten(),
    interpreter.get_tensor(14).flatten(),
)
# why is layer 2 output == layer 4 output (tensor 12 == tensor 14)?
# Because get_tensor cannot be used to get intermediate results (non-output layers)!
# To solve, must use experimental_preserve_all_tensors=True in tflite interpreter constructor
