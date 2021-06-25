# The aim is to mimic keras' dense layer

import os
from pathlib import Path
from numpy.lib.npyio import save

from tensorflow.python.training.tracking import base

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_layers
import tflite_runner
import utils

SEED = 0

tf.random.set_seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

# Setup the base model
tf.random.set_seed(SEED)

base_model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
        keras.layers.Dense(10),
        keras.layers.Dense(10),
    ]
)
base_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

saved_weights_path = f"saved_weights/compare_dense_{SEED}"
if not Path(saved_weights_path + ".index").exists():
    base_model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1, verbose=1
    )
    base_model.save_weights(saved_weights_path)
else:
    base_model.load_weights(
        saved_weights_path
    ).assert_existing_objects_matched().expect_partial()

# Setup the custom dense layer model
custom_model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Flatten(),
        custom_layers.DenseFakeQuant(10),
        custom_layers.DenseFakeQuant(10),
        custom_layers.DenseFakeQuant(10),
    ]
)
custom_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Copy weights from regular model to custom model,
# since custom model has issues training with the fake_quant thigns
for regular_layer, custom_layer in zip(base_model.layers, custom_model.layers):
    custom_layer.set_weights(regular_layer.get_weights())
    # Verify that weights were loaded
    for i, j in zip(regular_layer.get_weights(), custom_layer.get_weights()):
        assert np.array_equal(i, j)

_, base_model_accuracy = base_model.evaluate(test_images, test_labels, verbose=0)
_, custom_model_accuracy = custom_model.evaluate(test_images, test_labels, verbose=0)

quantized_tflite_model = tflite_runner.create_tflite_model(train_images, base_model)

# Evaluate and see if accuracy from TensorFlow persists to TFLite.
tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
    quantized_tflite_model, test_images, test_labels
)

print()
print("Base test accuracy:", base_model_accuracy)
print("Custom test accuracy:", custom_model_accuracy)
print("TFLite test_accuracy:", tflite_model_accuracy)

# Run test dataset on custom, and TFLite models
base_output: np.ndarray = base_model.predict(test_images)
base_output = base_output.flatten()
custom_output: np.ndarray = custom_model.predict(test_images)
custom_output = custom_output.flatten()
tflite_output = tflite_runner.run_tflite_model(quantized_tflite_model, test_images)
tflite_output = tflite_output.flatten()

# Check that Custom model is closer to tflite, than base model
# Also compare the custom fake quant model to tflite model
# TODO also compare to QATmodel?
utils.output_stats(base_output, tflite_output, "Base vs TFLite", "Dense", 1e-2, SEED)

utils.output_stats(
    custom_output, tflite_output, "Custom vs TFLite", "Dense", 1e-2, SEED
)
