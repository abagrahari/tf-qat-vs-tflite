import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras_lmu
import nengo_edge
import numpy as np
import tensorflow as tf
from custom_layers import calculate_min_max_from_tflite, calculate_scale_zp_from_min_max

import utils


def adjust_params(min, max):
    """Adjust quantization parameters as tflite does"""
    return [
        t.numpy()
        for t in calculate_min_max_from_tflite(*calculate_scale_zp_from_min_max(min, max))
    ]


def get_fq_weights(weights):
    """Symmetric / narrow range fake_quant"""
    return tf.quantization.fake_quant_with_min_max_args(
        weights,
        min(np.min(weights), -np.max(weights)),
        max(np.max(weights), -np.min(weights)),
        narrow_range=True,
    )


tf.random.set_seed(3)
rng = np.random.RandomState(3)

INPUT_D = 4
TIMESTEPS = 1
MEMORY_D = 10

inputs = rng.uniform(-0.5, 0.5, size=(320, TIMESTEPS, INPUT_D))
# 320 different inputs/sequences, of TIMESTEPS timesteps, and INPUT_D dimensions/features in each input

rnn_kernel = rng.uniform(-1, 1, size=(4, 10))
rnn_recurrent = rng.uniform(-1, 1, size=(10, 10))
hidden_kernel = rng.uniform(-1, 1, size=(40 * MEMORY_D, 10))
hidden_recurrent = rng.uniform(-1, 1, size=(10, 10))
dense_kernel = rng.uniform(-1, 1, size=(10, 10))

# Following weights copied from tflite model through netron.
add_op_input = np.array(
    [
        [
            -1.7126482725143433,
            -2.494635581970215,
            1.8153733015060425,
            -0.6371175646781921,
            1.781611680984497,
            -0.12725234031677246,
            -0.7038969993591309,
            -3.5249428749084473,
            2.6959800720214844,
            0.3533768057823181,
        ]
    ]
)
##################################################
# Manual computation - mimicking tflite
##################################################
# Similar approach to compare_manual_w_tflite.py

# Run fp32 forward pass
strided_slice_outputs = []
fc1_outputs = []
add_relu_outputs = []
fc_dense_outputs = []

for input in inputs:
    # run on all inputs
    input = np.expand_dims(input, axis=0)
    strided_slice_output = tf.strided_slice(
        input,
        # Settings from tflite model
        begin=[0, 0, 0],
        end=[0, 1, 4],
        strides=[1, 1, 1],
        begin_mask=5,
        ellipsis_mask=0,
        end_mask=5,
        new_axis_mask=0,
        shrink_axis_mask=2,
    ).numpy()
    strided_slice_outputs.append(strided_slice_output)
    # FC1
    x = tf.matmul(strided_slice_output, rnn_kernel)
    fc1_outputs.append(x)
    # Add with ReLU
    x = tf.math.add(x, add_op_input)
    x = tf.nn.relu(x)
    add_relu_outputs.append(x)
    # FC (Dense layer)
    x = tf.matmul(x, dense_kernel)
    fc_dense_outputs.append(x)

# Run quantized forward pass
rnn_kernel_quant = get_fq_weights(rnn_kernel)
dense_kernel_quant = get_fq_weights(dense_kernel)
# Inputs to add ops are quantized asymmetrically
p = adjust_params(np.min(add_op_input), np.max(add_op_input))
add_op_weights_quant = tf.quantization.fake_quant_with_min_max_args(add_op_input, p[0], p[1])

print("input params", *calculate_scale_zp_from_min_max(np.min(inputs), np.max(inputs)))
print(
    "strided_slice_outputs",
    *calculate_scale_zp_from_min_max(
        np.min(strided_slice_outputs), np.max(strided_slice_outputs)
    )
)
print(
    "fc1_outputs", *calculate_scale_zp_from_min_max(np.min(fc1_outputs), np.max(fc1_outputs))
)
print(
    "add_relu_outputs",
    *calculate_scale_zp_from_min_max(np.min(add_relu_outputs), np.max(add_relu_outputs))
)
print(
    "fc_dense_outputs",
    *calculate_scale_zp_from_min_max(np.min(fc_dense_outputs), np.max(fc_dense_outputs))
)

manual_outputs = []
# run on all inputs
for input in inputs:
    # Quantize inputs
    p = adjust_params(np.min(inputs), np.max(inputs))
    input = tf.quantization.fake_quant_with_min_max_args(input, p[0], p[1])
    input = np.expand_dims(input, axis=0)
    # StridedSlice op
    strided_slice_output = tf.strided_slice(
        input,
        # Settings from tflite model
        begin=[0, 0, 0],
        end=[0, 1, 4],
        strides=[1, 1, 1],
        begin_mask=5,
        ellipsis_mask=0,
        end_mask=5,
        new_axis_mask=0,
        shrink_axis_mask=2,
    ).numpy()

    p = adjust_params(np.min(strided_slice_outputs), np.max(strided_slice_outputs))
    strided_slice_output = tf.quantization.fake_quant_with_min_max_args(
        strided_slice_output, p[0], p[1]
    )
    # FC1 (lmu_kernel matches weights in tflite)
    x = tf.matmul(strided_slice_output, rnn_kernel_quant)
    p = adjust_params(np.min(fc1_outputs), np.max(fc1_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    # Add with ReLU
    x = tf.math.add(x, add_op_weights_quant)
    x = tf.nn.relu(x)
    p = adjust_params(np.min(add_relu_outputs), np.max(add_relu_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    # FC (Dense layer)
    x = tf.matmul(x, dense_kernel_quant)
    p = adjust_params(np.min(fc_dense_outputs), np.max(fc_dense_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    manual_outputs.append(x)

##################################################
# tflite computation
##################################################
# Define keras model with an RNN
inp = tf.keras.Input((TIMESTEPS, INPUT_D))
x = nengo_edge.layers.RNN(
    tf.keras.layers.SimpleRNNCell(
        10,
        activation="relu",
        kernel_initializer=tf.initializers.constant(rnn_kernel),
        recurrent_initializer=tf.initializers.constant(rnn_recurrent),
    )
)
x = x(inp, initial_state=[tf.ones((1, 10))])
x = tf.keras.layers.Dense(
    10,
    activation="linear",
    use_bias=False,
    kernel_initializer=tf.initializers.constant(dense_kernel),
)(x)
model = tf.keras.Model(inp, x)

model_output = model.predict(inputs, batch_size=1)


def representative_dataset():
    for i in range(inputs.shape[0]):
        yield [inputs[[i]].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()

if not os.path.exists("saved_models"):
    os.makedirs("saved_models")
with open("saved_models/simple_rnn.tflite", "wb") as f:
    f.write(tflite_model)

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

tflite_output = []
for input in inputs:
    input_scale, input_zero_point = input_details["quantization"]
    input = np.round(input / input_scale + input_zero_point).astype(np.uint8)
    input = np.expand_dims(input, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    output_scale, output_zero_point = output_details["quantization"]
    output = (output.astype(np.float32) - output_zero_point) * output_scale
    tflite_output.append(output)
tflite_output = np.array(tflite_output)

# Compare outputs
fc_dense_outputs = np.array(fc_dense_outputs).flatten()
manual_outputs = np.array(manual_outputs).flatten()
model_output = np.array(model_output).flatten()
tflite_output = np.array(tflite_output).flatten()
utils.output_stats(model_output, tflite_output, "Keras model vs tflite", 1e-2, 0)
utils.output_stats(fc_dense_outputs, model_output, "Manual LMU vs Keras Model", 1e-2, 0)
utils.output_stats(manual_outputs, tflite_output, "Manual FQ LMU vs tflite", 1e-2, 0)
