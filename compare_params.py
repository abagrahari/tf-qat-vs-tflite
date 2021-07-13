# Compare QAT and tflite quantization parameters

import argparse
import os
from pathlib import Path

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
from custom_layers import calculate_min_max_from_tflite

parser = argparse.ArgumentParser()

parser.add_argument("--seed", help="seed for tf.random", type=int, default=0)
parser.add_argument("--no-bias", help="seed for tf.random", action="store_false")
args = parser.parse_args()

SEED: int = args.seed
USE_BIAS: bool = args.no_bias  # Defaults to true

utils.remove_path("saved_models")
utils.remove_path("saved_weights")

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()


base_model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(10, use_bias=USE_BIAS),
        keras.layers.Dense(10, use_bias=USE_BIAS),
        keras.layers.Dense(10, use_bias=USE_BIAS),
        keras.layers.Dense(10, use_bias=USE_BIAS),
    ]
)
base_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train the model and save weights of the base model
saved_weights_path = f"saved_weights/compare_params_{SEED}"
if not Path(saved_weights_path + ".index").exists():
    base_model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1, verbose=1
    )
    base_model.save_weights(saved_weights_path)
else:
    base_model.load_weights(
        saved_weights_path
    ).assert_existing_objects_matched().expect_partial()


# MonkeyPatch to use AllValuesQuantizer instead of moving average
# to match behaviour of TFLite representative dataset quantization
default_8bit_quantize_registry.quantizers.MovingAverageQuantizer = (
    tfmot.quantization.keras.quantizers.AllValuesQuantizer
)

qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Manually calibrate QAT model
qat_model(train_images, training=True)

# Load weights to eliminate any possible changes from QAT
base_model.load_weights(
    saved_weights_path
).assert_existing_objects_matched().expect_partial()

# Create quantized model for TFLite from the base model
tflite_model = tflite_runner.create_tflite_model(
    train_images, base_model, f"saved_models/param_compare_{SEED}.tflite"
)

interpreter = tflite_runner.get_interpreter(tflite_model)
tensor_details = interpreter.get_tensor_details()

_, base_model_accuracy = base_model.evaluate(test_images, test_labels, verbose=0)
_, qat_model_accuracy = qat_model.evaluate(test_images, test_labels, verbose=0)
tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
    tflite_model, test_images, test_labels
)

print("Base test accuracy:", base_model_accuracy)
print("QAT test accuracy:", qat_model_accuracy)
print("TFLite test accuracy:", tflite_model_accuracy)

# Run test dataset on models
base_output: np.ndarray = base_model.predict(test_images)
qat_output: np.ndarray = qat_model.predict(test_images)
tflite_output = tflite_runner.run_tflite_model(tflite_model, test_images)

base_output = base_output.flatten()
qat_output = qat_output.flatten()
tflite_output = tflite_output.flatten()

# Determine if custom model is closer to tflite than QAT model:
utils.output_stats(qat_output, tflite_output, "QAT vs Base TFLite", 1e-2, SEED)

print("\nNote: the `Dense2` layer is the third Dense layer")
print("\nParameters directly from QAT model")
for weight in qat_model.weights:
    if "min" in weight.name or "max" in weight.name:
        print(f"{weight.name[:-2]:35} {weight.numpy():>15.6f}")


interpreter = tflite_runner.get_interpreter(tflite_model)
tensor_details = interpreter.get_tensor_details()

tflite_params = [{}, {}, {}, {}, {}]
if USE_BIAS:
    tflite_params[0]["input_scale"] = tensor_details[0]["quantization"][0]
    tflite_params[0]["input_zp"] = tensor_details[0]["quantization"][1]
    tflite_params[0]["output_scale"] = tensor_details[10]["quantization"][0]
    tflite_params[0]["output_zp"] = tensor_details[10]["quantization"][1]
    tflite_params[1]["input_scale"] = tensor_details[10]["quantization"][0]
    tflite_params[1]["input_zp"] = tensor_details[10]["quantization"][1]
    tflite_params[1]["kernel_scale"] = tensor_details[2]["quantization"][0]
    tflite_params[1]["kernel_zp"] = tensor_details[2]["quantization"][1]
    tflite_params[1]["bias_scale"] = tensor_details[3]["quantization"][0]
    tflite_params[1]["bias_zp"] = tensor_details[3]["quantization"][1]
    tflite_params[1]["output_scale"] = tensor_details[11]["quantization"][0]
    tflite_params[1]["output_zp"] = tensor_details[11]["quantization"][1]
    tflite_params[2]["input_scale"] = tensor_details[11]["quantization"][0]
    tflite_params[2]["input_zp"] = tensor_details[11]["quantization"][1]
    tflite_params[2]["kernel_scale"] = tensor_details[4]["quantization"][0]
    tflite_params[2]["kernel_zp"] = tensor_details[4]["quantization"][1]
    tflite_params[2]["bias_scale"] = tensor_details[5]["quantization"][0]
    tflite_params[2]["bias_zp"] = tensor_details[5]["quantization"][1]
    tflite_params[2]["output_scale"] = tensor_details[12]["quantization"][0]
    tflite_params[2]["output_zp"] = tensor_details[12]["quantization"][1]
    tflite_params[3]["input_scale"] = tensor_details[12]["quantization"][0]
    tflite_params[3]["input_zp"] = tensor_details[12]["quantization"][1]
    tflite_params[3]["kernel_scale"] = tensor_details[6]["quantization"][0]
    tflite_params[3]["kernel_zp"] = tensor_details[6]["quantization"][1]
    tflite_params[3]["bias_scale"] = tensor_details[7]["quantization"][0]
    tflite_params[3]["bias_zp"] = tensor_details[7]["quantization"][1]
    tflite_params[3]["output_scale"] = tensor_details[13]["quantization"][0]
    tflite_params[3]["output_zp"] = tensor_details[13]["quantization"][1]
    tflite_params[4]["input_scale"] = tensor_details[13]["quantization"][0]
    tflite_params[4]["input_zp"] = tensor_details[13]["quantization"][1]
    tflite_params[4]["kernel_scale"] = tensor_details[8]["quantization"][0]
    tflite_params[4]["kernel_zp"] = tensor_details[8]["quantization"][1]
    tflite_params[4]["bias_scale"] = tensor_details[9]["quantization"][0]
    tflite_params[4]["bias_zp"] = tensor_details[9]["quantization"][1]
    tflite_params[4]["output_scale"] = tensor_details[14]["quantization"][0]
    tflite_params[4]["output_zp"] = tensor_details[14]["quantization"][1]
else:
    # Numbering of tensors is different than above
    tflite_params[0]["input_scale"] = tensor_details[0]["quantization"][0]
    tflite_params[0]["input_zp"] = tensor_details[0]["quantization"][1]
    tflite_params[0]["output_scale"] = tensor_details[6]["quantization"][0]
    tflite_params[0]["output_zp"] = tensor_details[6]["quantization"][1]
    tflite_params[1]["input_scale"] = tensor_details[6]["quantization"][0]
    tflite_params[1]["input_zp"] = tensor_details[6]["quantization"][1]
    tflite_params[1]["kernel_scale"] = tensor_details[2]["quantization"][0]
    tflite_params[1]["kernel_zp"] = tensor_details[2]["quantization"][1]
    tflite_params[1]["output_scale"] = tensor_details[7]["quantization"][0]
    tflite_params[1]["output_zp"] = tensor_details[7]["quantization"][1]
    tflite_params[2]["input_scale"] = tensor_details[7]["quantization"][0]
    tflite_params[2]["input_zp"] = tensor_details[7]["quantization"][1]
    tflite_params[2]["kernel_scale"] = tensor_details[3]["quantization"][0]
    tflite_params[2]["kernel_zp"] = tensor_details[3]["quantization"][1]
    tflite_params[2]["output_scale"] = tensor_details[8]["quantization"][0]
    tflite_params[2]["output_zp"] = tensor_details[8]["quantization"][1]
    tflite_params[3]["input_scale"] = tensor_details[8]["quantization"][0]
    tflite_params[3]["input_zp"] = tensor_details[8]["quantization"][1]
    tflite_params[3]["kernel_scale"] = tensor_details[4]["quantization"][0]
    tflite_params[3]["kernel_zp"] = tensor_details[4]["quantization"][1]
    tflite_params[3]["output_scale"] = tensor_details[9]["quantization"][0]
    tflite_params[3]["output_zp"] = tensor_details[9]["quantization"][1]
    tflite_params[4]["input_scale"] = tensor_details[9]["quantization"][0]
    tflite_params[4]["input_zp"] = tensor_details[9]["quantization"][1]
    tflite_params[4]["kernel_scale"] = tensor_details[5]["quantization"][0]
    tflite_params[4]["kernel_zp"] = tensor_details[5]["quantization"][1]
    tflite_params[4]["output_scale"] = tensor_details[10]["quantization"][0]
    tflite_params[4]["output_zp"] = tensor_details[10]["quantization"][1]

# Print input quantization param
print(
    "\nParameters from tflite model, calculated using our scale/zp --> min/max implementation"
)

min, max = calculate_min_max_from_tflite(
    tflite_params[0]["input_scale"], tflite_params[0]["input_zp"]
)
param = "flatten/input_layer_min"
print(f"{param:35} {min.numpy():>15.6f}")
param = "flatten/input_layer_max"
print(f"{param:35} {max.numpy():>15.6f}")
# Print model quantization param
for i, layer in enumerate(tflite_params):
    if i == 0:
        continue
    name = ""
    if i == 1:
        name = "dense"
    else:
        name = f"dense_{i-1}"
    for calc in ["kernel", "output"]:
        min, max = calculate_min_max_from_tflite(
            layer[f"{calc}_scale"], layer[f"{calc}_zp"]
        )
        if calc == "output":
            calc = "post_activation"  # Match QAT print statement
        param = f"{name}/{calc}_min"
        print(f"{param:35} {min.numpy():>15.6f}")
        param = f"{name}/{calc}_max"
        print(f"{param:35} {max.numpy():>15.6f}")

if USE_BIAS:
    print(
        "\nParameters from tflite model, for quantization of bias calculated using our scale/zp --> min/max implementation. (QAT does not FakeQuant bias)"
    )
    for i, layer in enumerate(tflite_params):
        if i == 0:
            continue
        name = ""
        if i == 1:
            name = "dense"
        else:
            name = f"dense_{i-1}"
        min, max = calculate_min_max_from_tflite(layer["bias_scale"], layer["bias_zp"])
        param = f"{name}/bias_min"
        print(f"{param:35} {min.numpy():>15.6f}")
        param = f"{name}/bias_max"
        print(f"{param:35} {max.numpy():>15.6f}")
