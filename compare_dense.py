# The aim is to mimic keras' dense layer and TFLite's quantization approach.

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras

import custom_layers
import tflite_runner
import utils

parser = argparse.ArgumentParser()

parser.add_argument("--seed", help="seed for tf.random", type=int, default=0)
args = parser.parse_args()

SEED: int = args.seed

tf.random.set_seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

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

tflite_model = tflite_runner.create_tflite_model(
    train_images, base_model, f"saved_models/base_model_dense_{SEED}.tflite"
)

# Setup the custom dense layer model with params from tflite
tensor_details = tflite_runner.get_interpreter(tflite_model).get_tensor_details()

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
        ),
    ]
)
custom_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Copy weights from regular model to custom model,
# since we focus on using custom model for inference, and don't train it
# Also, the custom model may have issues training with the fake_quant things
for regular_layer, custom_layer in zip(base_model.layers, custom_model.layers):
    custom_layer.set_weights(regular_layer.get_weights())
    # Verify that weights were loaded
    for i, j in zip(regular_layer.get_weights(), custom_layer.get_weights()):
        assert np.array_equal(i, j)

_, base_model_accuracy = base_model.evaluate(test_images, test_labels, verbose=0)
_, custom_model_accuracy = custom_model.evaluate(test_images, test_labels, verbose=0)
tflite_model_accuracy = tflite_runner.evaluate_tflite_model(
    tflite_model, test_images, test_labels
)

print("Base test accuracy:", base_model_accuracy)
print("Custom test accuracy:", custom_model_accuracy)
print("TFLite test_accuracy:", tflite_model_accuracy)

# index is 0->4 (Flatten,Dense,Dense,Dense,Dense)
extractor = keras.Model(
    inputs=custom_model.inputs, outputs=[layer.output for layer in custom_model.layers]
)
# list of tf.Tensors. 1 element for each layer
extractor_output = extractor(test_images)
extractor_tflite_outputs = tflite_runner.collect_intermediate_outputs(
    tflite_model, test_images
)
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 5)
for idx, (intermediate_output, intermediate_tflite_output) in enumerate(
    zip(extractor_output, extractor_tflite_outputs)
):
    custom_output = intermediate_output.numpy().flatten()
    tflite_output = intermediate_tflite_output.numpy().flatten()
    utils.output_stats(
        custom_output,
        tflite_output,
        f"Layer {idx}",
        1e-2,
        SEED,
        axs[idx],
    )
plt.show()


# Run test dataset on models
base_output: np.ndarray = base_model.predict(test_images)
custom_output: np.ndarray = custom_model.predict(test_images)
tflite_output = tflite_runner.run_tflite_model(tflite_model, test_images)
base_output = base_output.flatten()
custom_output = custom_output.flatten()
tflite_output = tflite_output.flatten()

# Check that Custom model is closer to tflite, than base model
utils.output_stats(
    custom_output, tflite_output, "Custom vs TFLite - Overall", 1e-2, SEED
)

# comparision = np.isclose(custom_output, tflite_output, rtol=0, atol=1e-2)
# if np.count_nonzero(~comparision) != 0:
#     print("Differing model ouputs are:")
#     for i, val in enumerate(comparision):
#         if val == False:
#             print("CustomTF:\t", custom_output[i], "\tTFLite:\t", tflite_output[i])
