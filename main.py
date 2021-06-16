# End-to-end example for quantization aware training + tflite

# Here we:
# - Train a tf.keras model for MNIST from scratch.
# - Fine tune the model by applying the quantization aware training API, see the accuracy, and export a quantization aware model.
# - Use the model to create an actually quantized model for the TFLite backend.
# - See the persistence of accuracy in the TFLite model.

# The aim is to check for equivalence between a QAT model and its TFLite model

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
MODEL_TYPE = "CNN"  # Set to 'CNN' or 'dense3' or 'dense4' etc.

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

import utils

# Set seed for reproducibility
tf.random.set_seed(4)

# Train a model for MNIST without quantization aware training

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
def get_model(model_type: str):
    """Options: "dense#', or 'cnn'."""
    layers = [keras.layers.InputLayer(input_shape=(28, 28))]
    if "dense" in model_type:
        # Model is n stacked Dense layers
        layers.append(keras.layers.Flatten())
        layers.extend(
            keras.layers.Dense(10) for _ in range(int(model_type.split("dense")[1]))
        )
    else:
        # Model is a CNN
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

model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_split=0.1,
)


# Clone and fine-tune pre-trained model with quantization aware training
# We QAT to the whole model and can see this in the model summary. All layers are now prefixed by "quant".
# Note: the resulting model is quantization aware but not quantized (e.g. the weights are float32 instead of int8).
# The sections after show how to create a quantized model from the quantization aware one, using the TFLiteConverter

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

# Train and evaluate the model against baseline
# To demonstrate fine tuning after training the model for just an epoch, fine tune with quantization aware training on a subset of the training data.

train_images_subset = train_images[0:1000]  # out of 60000
train_labels_subset = train_labels[0:1000]

qat_model.fit(
    train_images_subset,
    train_labels_subset,
    batch_size=500,
    epochs=1,
    validation_split=0.1,
)

# there is minimal to no loss in test accuracy after quantization aware training, compared to the baseline.
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)
_, qat_model_accuracy = qat_model.evaluate(test_images, test_labels, verbose=0)

# Create quantized model for TFLite backend, using 'Dynamic range quantization'
# After this, we have an actually quantized model with int8 weights and uint8 activations.
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# TF's QAT example uses Dynamic range quantization

# Converter options for all INT8 conversion:
# TODO try using representive_dataset for full integer quantization
# def representative_dataset():
#     for data in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
#         yield [data]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8 for Coral Edge TPU
# converter.inference_output_type = tf.int8  # or tf.uint8 for Coral Edge TPU
# converter.representative_dataset = representative_dataset
quantized_tflite_model = converter.convert()

# Evaluate the quantized model and see that the accuracy from TensorFlow persists to the TFLite backend.
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

# input_type = interpreter.get_input_details()[0]["dtype"]
# print("tflite input: ", input_type)
# output_type = interpreter.get_output_details()[0]["dtype"]
# print("tflite output: ", output_type)


def run_tflite_model(interpreter: tf.lite.Interpreter):
    """Helper function to return outputs on the test dataset using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    outputs = []
    for test_image in test_images:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension
        outputs.append(interpreter.get_tensor(output_index)[0])

    return np.array(outputs)


def evaluate_model(interpreter):
    """Helper function to evaluate the TF Lite model on the test dataset."""

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    outputs = run_tflite_model(interpreter)
    for output in outputs:
        digit = np.argmax(output)
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy


tflite_model_accuracy = evaluate_model(interpreter)

print("Baseline test accuracy:", baseline_model_accuracy)
print("QAT test accuracy:", qat_model_accuracy)
print("TFLite test_accuracy:", tflite_model_accuracy)

# Run on baseline, QAT, and TFLite models
base_output = model.predict(test_images)
qat_output = qat_model.predict(test_images)
tflite_output = run_tflite_model(interpreter)

base_output = base_output.flatten()
qat_output = qat_output.flatten()
tflite_output = tflite_output.flatten()

utils.output_stats(base_output, qat_output, "Base vs QAT", MODEL_TYPE, 1e-2)
utils.output_stats(base_output, tflite_output, "Base vs TFLite", MODEL_TYPE, 1e-2)
utils.output_stats(qat_output, tflite_output, "QAT vs TFLite", MODEL_TYPE, 1e-2)
