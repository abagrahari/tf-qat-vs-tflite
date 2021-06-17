# End-to-end example for quantization aware training + tflite
# The aim is to check for equivalence between a QAT model and its TFLite model

# Here we:
# - Train a tf.keras model for MNIST from scratch.
# - Fine tune the model by applying the quantization aware training API, and create a quantization aware model.
# - Use the model to create an actually-quantized model for TFLite.
# - See the persistence of accuracy in the TFLite model.
# - Compare the error in raw output values.


import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="Model type",
    choices=["CNN", "dense1", "dense2", "dense3", "dense4"],
    default="CNN",
)
parser.add_argument("--seed", help="seed for tf.random", type=int, default=3)
parser.add_argument(
    "--quantize", help="Quantize TFLite model to 8 bit", action="store_true"
)
args = parser.parse_args()


MODEL_TYPE = args.model
QUANTIZE_TO_8BIT = args.quantize
SEED = args.seed

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

import utils

# Set seed for reproducibility
tf.random.set_seed(SEED)

# Train a model for MNIST without quantization aware training

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
def get_model(model_type: str):
    """Options: "dense#', or 'CNN'."""
    layers = [keras.layers.InputLayer(input_shape=(28, 28))]
    if "dense" in model_type:
        # Model is <n> stacked Dense layers
        layers.append(keras.layers.Flatten())
        layers.extend(
            keras.layers.Dense(10, activation="linear")
            for _ in range(int(model_type.split("dense")[1]))
        )
    else:
        # Use CNN for the Model
        layers.extend(
            [
                keras.layers.Reshape(target_shape=(28, 28, 1)),
                keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(10),
            ]
        )
    return keras.Sequential(layers)


model = get_model(MODEL_TYPE)

# Train the digit classification model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(train_images, train_labels, epochs=1, validation_split=0.1, verbose=1)

# Clone and fine-tune the regularaly trained model with quantization aware training
# We apply QAT to the whole model and can see this in the model summary.
# All layers are now prefixed by "quant".
# Note: the resulting model is quantization aware but not quantized (e.g. the
# weights are float32 instead of int8).
# The sections after will create a quantized model from the quantization aware one,
# using the TFLiteConverter

quantize_model = tfmot.quantization.keras.quantize_model

# quantization aware model
qat_model = quantize_model(model)

# `quantize_model` requires a recompile.
qat_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# qat_model.summary()

# To demonstrate fine tuning after training the model for just an epoch,
# fine tune with quantization aware training on a subset of the training data.
train_images_subset = train_images[0:1000]  # out of 60000
train_labels_subset = train_labels[0:1000]
qat_model.fit(
    train_images_subset,
    train_labels_subset,
    batch_size=500,
    epochs=1,
    validation_split=0.1,
    verbose=1,
)

# We'll see minimal to no loss in test accuracy after quantization aware training, compared to the baseline.
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
_, qat_model_accuracy = qat_model.evaluate(test_images, test_labels, verbose=0)

# Create quantized model for TFLite backend
# After this, we have an actually quantized model with int8 weights and uint8 activations.
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# TF's QAT example uses Dynamic range quantization
# These settings are used above.


def representative_dataset():
    for data in (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(train_images.shape[0])
        .batch(1)
        .take(1000)
    ):
        yield [tf.dtypes.cast(data, tf.float32)]


if QUANTIZE_TO_8BIT:
    # For all INT8 conversion, we need some additional converter settings:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8 for Coral Edge TPU
    converter.inference_output_type = tf.int8  # or tf.uint8 for Coral Edge TPU
    converter.representative_dataset = representative_dataset
quantized_tflite_model = converter.convert()


def run_tflite_model(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    """Helper function to return outputs on the test dataset using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Run predictions on every image in the "test" dataset.
    outputs = []
    for test_image in test_images:

        # Check if the input type is quantized, then rescale input data to uint8
        # as shown in TF's Post-Training Integer Quantization Example
        if input_details["dtype"] in [np.uint8, np.int8]:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point
            # The TensorFlow example does not have the np.round() op shown below.
            # However during testing, without it, values like `125.99998498`
            # are replaced with 125 instead of 126, since we would directly
            # cast to int8/uint8
            test_image = np.round(test_image)

        # Pre-processing: add batch dimension and convert to datatype to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and dequantize the output
        # based on TensorFlow's quantization params
        # We dequantize the outputs so we can directly compare the raw
        # outputs with the QAT model
        output = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] in [np.uint8, np.int8]:
            output_scale, output_zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        outputs.append(output)

    return np.array(outputs)


def evaluate_model(tflite_model):
    """Helper function to evaluate the TF Lite model on the test dataset."""

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    outputs = run_tflite_model(tflite_model)
    for output in outputs:
        digit = np.argmax(output)
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy


# Evaluate the quantized model and see if accuracy from TensorFlow
# persists to TFLite.
tflite_model_accuracy = evaluate_model(quantized_tflite_model)

print()
print("Baseline test accuracy:", baseline_model_accuracy)
print("QAT test accuracy:", qat_model_accuracy)
print("TFLite test_accuracy:", tflite_model_accuracy)
print()

# Run test dataset on baseline, QAT, and TFLite models
baseline_output = model.predict(test_images)
qat_output = qat_model.predict(test_images)
tflite_output = run_tflite_model(quantized_tflite_model)

baseline_output = baseline_output.flatten()
qat_output = qat_output.flatten()
tflite_output = tflite_output.flatten()

utils.output_stats(
    baseline_output, qat_output, "Baseline vs QAT", MODEL_TYPE, 1e-2, SEED
)
utils.output_stats(
    baseline_output, tflite_output, "Baseline vs TFLite", MODEL_TYPE, 1e-2, SEED
)
utils.output_stats(
    qat_output, tflite_output, "QAT vs TFLite", MODEL_TYPE, 1e-2, SEED, True
)
