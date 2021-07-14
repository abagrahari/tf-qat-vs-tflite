# The aim is to mimic keras' dense layer and TFLite's quantization approach.

import argparse
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
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
interpreter = tflite_runner.get_interpreter(tflite_model)
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


def run_intermeidate_outputs(lower_limit, upper_limit):
    imgs = test_images[lower_limit:upper_limit]
    if lower_limit == upper_limit:
        # Expand first dimension to account for batch size when running single image
        imgs = np.expand_dims(test_images[lower_limit], axis=0).astype(np.float32)
        assert imgs.shape == (1, 28, 28)
    extractor_output = extractor(imgs)
    extractor_tflite_outputs = tflite_runner.collect_intermediate_outputs(
        tflite_model, imgs
    )
    return extractor_output, extractor_tflite_outputs


def binary_search(imgs, low, high, relative_error_target):
    """Recursive Binary Search.

    Returns: index of image that causes the given relative_error, or -1
    """
    # Check base case
    if high >= low:
        mid = (high + low) // 2
        extractor_output, extractor_tflite_outputs = run_intermeidate_outputs(low, mid)
        custom_output = extractor_output[2].numpy().flatten()
        tflite_output = extractor_tflite_outputs[2].numpy().flatten()
        max_rel_err = utils.get_max_rel_err(custom_output, tflite_output, 1e-2)
        if max_rel_err == relative_error_target:
            if low == mid:
                # the image has been found
                return mid
            # Otherwise, the desired image is in the left half
            return binary_search(imgs, low, mid - 1, relative_error_target)

        if max_rel_err != relative_error_target:
            # the desired image must be in the right half
            return binary_search(imgs, mid, high, relative_error_target)
    else:
        # Element is not present in the array
        return -1


# To search for image that causes max relative error:
# result = binary_search(test_images, 0, test_images.shape[0], 80)
# if result != -1:
#     print("Index: ", result)

# To explore layer discrepencies for a single image:
# custom_model.layers[1].print_int8_layer_outputs = True
# custom_model.layers[2].print_int8_layer_outputs = True
# custom_model.layers[3].print_int8_layer_outputs = True
# custom_model.layers[4].print_int8_layer_outputs = True
# lists of tf.Tensors. 1 element for each layer
# extractor_output, extractor_tflite_outputs = run_intermeidate_outputs(9070, 9070)
# custom_model.layers[1].print_int8_layer_outputs = False
# custom_model.layers[2].print_int8_layer_outputs = False
# custom_model.layers[3].print_int8_layer_outputs = False
# custom_model.layers[4].print_int8_layer_outputs = False
extractor_output, extractor_tflite_outputs = run_intermeidate_outputs(
    0, test_images.shape[0]
)

# Compare layer by layer outputs
fig, axs = plt.subplots(1, 5)
for idx, (intermediate_output, intermediate_tflite_output) in enumerate(
    zip(extractor_output, extractor_tflite_outputs)
):
    custom_output = intermediate_output.numpy().flatten()
    tflite_output = intermediate_tflite_output.numpy().flatten()
    # utils.output_stats(
    #     custom_output,
    #     tflite_output,
    #     f"Layer {idx}",
    #     1e-2,
    #     SEED,
    #     axs[idx],
    # )

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

# plt.show()
