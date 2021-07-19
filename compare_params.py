# Comparing QAT and tflite quantization parameters

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
from custom_layers import (
    calculate_min_max_from_tflite,
    calculate_scale_zp_from_min_max,
    fake_quant,
)

# MonkeyPatch to use AllValuesQuantizer instead of moving average
# to match behaviour of TFLite representative dataset quantization
default_8bit_quantize_registry.quantizers.MovingAverageQuantizer = (
    tfmot.quantization.keras.quantizers.AllValuesQuantizer
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", help="seed for tf.random", type=int, default=0)
parser.add_argument(
    "--no-bias", help="Whether Dense layers should include bias", action="store_false"
)
args = parser.parse_args()

SEED: int = args.seed
USE_BIAS: bool = args.no_bias  # Defaults to true

tf.random.set_seed(SEED)
np.random.seed(SEED)

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = utils.load_mnist()

saved_weights_path = f"saved_weights/compare_params_{SEED}_{USE_BIAS}"

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
if not Path(saved_weights_path + ".index").exists():
    base_model.fit(
        train_images, train_labels, epochs=1, validation_split=0.1, verbose=1
    )
    base_model.save_weights(saved_weights_path)
else:
    base_model.load_weights(
        saved_weights_path
    ).assert_existing_objects_matched().expect_partial()


qat_model = tfmot.quantization.keras.quantize_model(base_model)
qat_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Calibrate QAT model
qat_model(train_images, training=True)

# Create quantized model for TFLite from the base model
tflite_model = tflite_runner.create_tflite_model(
    train_images, base_model, f"saved_models/param_compare_{SEED}_{USE_BIAS}.tflite"
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


print("\nNote: the `Dense2` layer is the third Dense layer")
print("\nParameters directly from QAT model")
for weight in qat_model.weights:
    if "min" in weight.name or "max" in weight.name:
        utils.print_formatted(weight.name[:-2], weight.numpy())


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
utils.print_formatted("flatten/input_layer_min", min.numpy())
utils.print_formatted("flatten/input_layer_max", max.numpy())
# Print tflite quantization params
for i, layer in enumerate(tflite_params):
    if i == 0:
        continue
    name = ""
    if i == 1:
        name = "dense"
    else:
        name = f"dense_{i-1}"
    for calc in ["kernel", "output"]:
        minspec = -127 if calc == "kernel" else -128
        min, max = calculate_min_max_from_tflite(
            layer[f"{calc}_scale"], layer[f"{calc}_zp"], min_spec=minspec
        )
        if calc != "kernel":
            # Verify that min/max was derived correctly from tflite params, by recalculating
            # the scale param.
            # Does not match up for kernel params, due to narrow_range used in kernel
            qat_paper_scale, _ = calculate_scale_zp_from_min_max(min, max)
            assert np.allclose(
                qat_paper_scale, layer[f"{calc}_scale"], rtol=0, atol=1e-5
            )
        if calc == "output":
            calc = "post_activation"  # Match QAT print statement
        utils.print_formatted(f"{name}/{calc}_min", min.numpy())
        utils.print_formatted(f"{name}/{calc}_max", max.numpy())

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
        utils.print_formatted(f"{name}/bias_min", min.numpy())
        utils.print_formatted(f"{name}/bias_max", max.numpy())


# Compare First dense layer outputs
# index is 0->5 (QuantizeLayer,Flatten,Dense,Dense,Dense,Dense)
extractor = keras.Model(inputs=qat_model.inputs, outputs=qat_model.layers[2].output)
extractor_output = extractor(train_images)
if USE_BIAS:
    extractor_tflite_outputs = tflite_runner.collect_intermediate_outputs(
        tflite_model, train_images
    )
else:

    extractor_tflite_outputs = tflite_runner.collect_intermediate_outputs_no_bias(
        tflite_model, train_images
    )
qat_output = extractor_output.numpy()
tflite_output = extractor_tflite_outputs[1].numpy()

print("\nParameters from intermediate outputs")
utils.print_formatted("QAT - dense/post_activation_min", np.min(qat_output))
utils.print_formatted("QAT - dense/post_activation_max", np.max(qat_output))
utils.print_formatted("tflite - dense/post_activation_min", np.min(tflite_output))
utils.print_formatted("tflite - dense/post_activation_max", np.max(tflite_output))


# Check which min/max is "correct". i.e. take all the mnist digits,
# multiply them by the first dense weight matrix, and check the min/max of the output.
# see if it matches tflite or qat (or neither)
# We can use the tflite scale/zp for fake_quantizing kernel and inputs, since they
# match with the QAT min/max params.
kernel = qat_model.weights[4]  # Get kernel from QAT model
# FakeQuant kernel based on params from tflite model. (min/max is same in QAT model)
fq_kernel = fake_quant(
    kernel,
    tflite_params[1]["kernel_scale"],
    tflite_params[1]["kernel_zp"],
    narrow=True,  # tflite spec says it uses narrow_range for weights, with below value
    min_spec=-127,
)
if not USE_BIAS:
    outputs = []
    for image in train_images:
        # Flatten image
        image = tf.cast(tf.reshape(image, [-1, 784]), tf.float32)
        assert image.shape == (1, 784)
        # Quantize the input (tflite and QAT have same min/max params)
        fq_input = fake_quant(
            image, tflite_params[0]["input_scale"], tflite_params[0]["input_zp"]
        )
        y: tf.Tensor = tf.matmul(fq_input, fq_kernel)
        assert y.shape == (1, 10)
        # no bias adddition
        # linear activation function

        # Not fakeQuantizing outputs, in order to compare min/max params

        outputs.append(y)

    outputs = np.array(outputs)

    # Print input quantization param
    print("\nParameters from manual computation")
    utils.print_formatted("dense/post_activation_min", np.min(outputs))
    utils.print_formatted("dense/post_activation_max", np.max(outputs))
