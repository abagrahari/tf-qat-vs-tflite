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

import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_registry,
)

import tflite_runner
import utils

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    help="Model type",
    choices=["CNN", "dense1", "dense2", "dense3", "dense4"],
    default="CNN",
)
parser.add_argument("--seed", help="seed for tf.random", type=int, default=3)
parser.add_argument(
    "--eval",
    help="Monkeypatch to AllValuesQuantizer and load weights from previous QAT model",
    action="store_true",
)
args = parser.parse_args()

MODEL_TYPE: str = args.model
SEED: int = args.seed
EVAL_PATCHED_QAT: bool = args.eval

tf.random.set_seed(SEED)

# Train a model for MNIST without QAT
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

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


saved_weights_path = f"saved_weights/{MODEL_TYPE}_{SEED}"
base_model = get_model(MODEL_TYPE)

if not EVAL_PATCHED_QAT:
    # Train the base model and save weights of the un-patched QAT model

    # Train the base model
    base_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    base_model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1, verbose=1
    )

    # Clone and fine-tune the regularaly trained model with quantization aware training
    # We apply QAT to the whole model and c
    # Note: the resulting model is quantization aware but not quantized (e.g. the
    # weights are float32 instead of int8).
    # We will create a quantized model from the quantization aware one using the TFLiteConverter
    qat_model = tfmot.quantization.keras.quantize_model(base_model)
    qat_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # fine tune with quantization aware training
    qat_model.fit(
        train_images,
        train_labels,
        batch_size=500,
        epochs=1,
        validation_split=0.1,
        verbose=1,
    )
    qat_model.save_weights(saved_weights_path)

elif EVAL_PATCHED_QAT:
    # Monkeypatch, load QAT model weights, run callibration, and compare tflite model

    # MonkeyPatch to use AllValuesQuantizer instead of moving average
    # to match behaviour of TFLite representative dataset quantization
    # (it weights all items in the representative dataset equally,
    # it doesn't do a moving average)
    # We monkey patch only in eval mode and not training mode, since during training
    # we want a MovingAverageQuantizer to remove effect of poor initial quantization values
    default_8bit_quantize_registry.quantizers.MovingAverageQuantizer = (
        tfmot.quantization.keras.quantizers.AllValuesQuantizer
    )

    qat_model2 = tfmot.quantization.keras.quantize_model(base_model)
    qat_model2.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    # Note: the loaded weights were not made with AllValuesQuantizer
    qat_model2.load_weights(
        saved_weights_path
    ).assert_existing_objects_matched().expect_partial()

    # calibrate QAT model, after the monkey patch
    qat_model2(train_images, training=True)

    _, qat_model_accuracy = qat_model2.evaluate(test_images, test_labels, verbose=0)

    # Create quantized model for TFLite
    # After this, we will have an actually quantized model with int8 weights and uint8 activations.

    quantized_tflite_model = tflite_runner.create_tflite_model(train_images, qat_model2)

    # Evaluate and see if accuracy from TensorFlow persists to TFLite.
    tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
        quantized_tflite_model, test_images, test_labels
    )

    print("QAT test accuracy:", qat_model_accuracy)
    print("TFLite test_accuracy:", tflite_model_accuracy)

    # Run test dataset on QAT, and TFLite models
    qat_output = qat_model2.predict(test_images)
    tflite_output = tflite_runner.run_tflite_model(quantized_tflite_model, test_images)

    qat_output = qat_output.flatten()
    tflite_output = tflite_output.flatten()

    utils.output_stats(
        qat_output, tflite_output, "QAT vs TFLite", MODEL_TYPE, 1e-2, SEED
    )
