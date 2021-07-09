# The aim is to compare a QAT model, its TFLite model,
# and a custom keras dense layer that mimics TFLite's quantization approach.

# Here we:
# - Train a tf.keras model for MNIST from scratch.
# - Fine tune the model by applying the quantization aware training API, and create a quantization aware model.
# - Use the model to create an actually-quantized model for TFLite.
# - Use the base model to create a custom-layered model and incorporate tflite model's quantization parameters
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

import custom_layers
import tflite_runner
import utils

parser = argparse.ArgumentParser()

parser.add_argument("--seed", help="seed for tf.random", type=int, default=0)
parser.add_argument(
    "--eval",
    help="Monkeypatch to AllValuesQuantizer and load weights from previous model",
    action="store_true",
)
args = parser.parse_args()

SEED: int = args.seed
EVAL: bool = args.eval

tf.random.set_seed(SEED)

# Train a model for MNIST without QAT
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

saved_weights_path = f"saved_weights/3compare_{SEED}"

base_model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10),
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

if not EVAL:
    # Train the model and save weights of the base and un-patched QAT model
    base_model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1, verbose=1
    )
    base_model.save_weights(saved_weights_path)

    # Clone and fine-tune the regularaly trained model with quantization aware training
    # We apply QAT to the whole model
    # Note: the resulting model is quantization aware but not quantized (e.g. the
    # weights are float32 instead of int8).
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
    qat_model.save_weights(saved_weights_path + "_qat")

elif EVAL:
    # Monkeypatch, load model weights, run callibration, and compare tflite model

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
        saved_weights_path + "_qat"
    ).assert_existing_objects_matched().expect_partial()

    # calibrate QAT model, after the monkey patch
    qat_model2(train_images, training=True)

    # Create quantized model for TFLite from the patched QAT model
    qat_tflite_model = tflite_runner.create_tflite_model(
        train_images, qat_model2, f"saved_models/3compare_qat_{SEED}.tflite"
    )

    # Setup the custom dense layer model with params from tflite
    # Load weights to eliminate any possible changes from QAT
    base_model.load_weights(
        saved_weights_path
    ).assert_existing_objects_matched().expect_partial()

    # Create quantized model for TFLite from the base model
    base_tflite_model = tflite_runner.create_tflite_model(
        train_images, base_model, f"saved_models/3compare_base_{SEED}.tflite"
    )

    interpreter = tflite_runner.get_interpreter(base_tflite_model)
    tensor_details = interpreter.get_tensor_details()

    custom_model = keras.Sequential(
        [
            custom_layers.FlattenTFLite(
                input_scale=tensor_details[0]["quantization"][0],
                input_zp=tensor_details[0]["quantization"][1],
                output_scale=tensor_details[10]["quantization"][0],
                output_zp=tensor_details[10]["quantization"][1],
                input_shape=(28, 28),
            ),
            custom_layers.DenseTFLite(
                10,
                input_scale=tensor_details[10]["quantization"][0],
                input_zp=tensor_details[10]["quantization"][1],
                kernel_scale=tensor_details[2]["quantization"][0],
                kernel_zp=tensor_details[2]["quantization"][1],
                bias_scale=tensor_details[3]["quantization"][0],
                bias_zp=tensor_details[3]["quantization"][1],
                output_scale=tensor_details[11]["quantization"][0],
                output_zp=tensor_details[11]["quantization"][1],
                kernel_tflite=interpreter.get_tensor(2).transpose(),
                bias_tflite=interpreter.get_tensor(3),
            ),
            custom_layers.DenseTFLite(
                10,
                input_scale=tensor_details[11]["quantization"][0],
                input_zp=tensor_details[11]["quantization"][1],
                kernel_scale=tensor_details[4]["quantization"][0],
                kernel_zp=tensor_details[4]["quantization"][1],
                bias_scale=tensor_details[5]["quantization"][0],
                bias_zp=tensor_details[5]["quantization"][1],
                output_scale=tensor_details[12]["quantization"][0],
                output_zp=tensor_details[12]["quantization"][1],
                kernel_tflite=interpreter.get_tensor(4).transpose(),
                bias_tflite=interpreter.get_tensor(5),
            ),
            custom_layers.DenseTFLite(
                10,
                input_scale=tensor_details[12]["quantization"][0],
                input_zp=tensor_details[12]["quantization"][1],
                kernel_scale=tensor_details[6]["quantization"][0],
                kernel_zp=tensor_details[6]["quantization"][1],
                bias_scale=tensor_details[7]["quantization"][0],
                bias_zp=tensor_details[7]["quantization"][1],
                output_scale=tensor_details[13]["quantization"][0],
                output_zp=tensor_details[13]["quantization"][1],
                kernel_tflite=interpreter.get_tensor(6).transpose(),
                bias_tflite=interpreter.get_tensor(7),
            ),
            custom_layers.DenseTFLite(
                10,
                input_scale=tensor_details[13]["quantization"][0],
                input_zp=tensor_details[13]["quantization"][1],
                kernel_scale=tensor_details[8]["quantization"][0],
                kernel_zp=tensor_details[8]["quantization"][1],
                bias_scale=tensor_details[9]["quantization"][0],
                bias_zp=tensor_details[9]["quantization"][1],
                output_scale=tensor_details[14]["quantization"][0],
                output_zp=tensor_details[14]["quantization"][1],
                kernel_tflite=interpreter.get_tensor(8).transpose(),
                bias_tflite=interpreter.get_tensor(9),
            ),
        ]
    )
    custom_model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    # Copy weights from regular base model to custom model,
    # since we focus on using custom model for inference, and don't train it
    for regular_layer, custom_layer in zip(base_model.layers, custom_model.layers):
        custom_layer.set_weights(regular_layer.get_weights())
        # Verify that weights were loaded
        for i, j in zip(regular_layer.get_weights(), custom_layer.get_weights()):
            assert np.array_equal(i, j)

    _, base_model_accuracy = base_model.evaluate(test_images, test_labels, verbose=0)
    _, qat_model_accuracy = qat_model2.evaluate(test_images, test_labels, verbose=0)
    _, custom_model_accuracy = custom_model.evaluate(
        test_images, test_labels, verbose=0
    )
    # custom layer's accuracy is lower than QAT model - perhaps because QAT model trains for 1-2 more epochs?
    base_tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
        base_tflite_model, test_images, test_labels
    )
    qat_tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
        qat_tflite_model, test_images, test_labels
    )

    print("Base test accuracy:", base_model_accuracy)
    print("QAT test accuracy:", qat_model_accuracy)
    print("Custom layer test accuracy:", custom_model_accuracy)
    print("Base TFLite test accuracy:", base_tflite_model_accuracy)
    print("QAT TFLite test_accuracy:", qat_tflite_model_accuracy)

    # Run test dataset on models
    base_output: np.ndarray = base_model.predict(test_images)
    custom_output: np.ndarray = custom_model.predict(test_images)
    qat_output: np.ndarray = qat_model2.predict(test_images)
    base_tflite_output = tflite_runner.run_tflite_model(base_tflite_model, test_images)
    qat_tflite_output = tflite_runner.run_tflite_model(qat_tflite_model, test_images)

    base_output = base_output.flatten()
    custom_output = custom_output.flatten()
    qat_output = qat_output.flatten()
    base_tflite_output = base_tflite_output.flatten()
    qat_tflite_output = qat_tflite_output.flatten()

    # Determine if custom model is closer to tflite than QAT model:
    utils.output_stats(qat_output, qat_tflite_output, "QAT vs QAT TFLite", 1e-2, SEED)
    utils.output_stats(
        custom_output, base_tflite_output, "Custom vs Base TFLite", 1e-2, SEED
    )
