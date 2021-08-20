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

lmu_kernel = rng.uniform(-1, 1, size=(4, MEMORY_D))
lmu_recurrent = rng.uniform(-1, 1, size=(1 * MEMORY_D, 4 * MEMORY_D))
hidden_kernel = rng.uniform(-1, 1, size=(4 * MEMORY_D, 10))
hidden_recurrent = rng.uniform(-1, 1, size=(10, 10))
dense_kernel = rng.uniform(-1, 1, size=(40, 10))

# Following weights copied from tflite model through netron.
batchmatmul_weights = np.array(
    [
        [0.20077991485595703],
        [-0.4733007252216339],
        [0.5095000863075256],
        [-0.12802784144878387],
    ]
).transpose()
add_op_1_input = np.array(
    [
        [
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
            [0.5212634801864624, 0.421031653881073, -0.15629586577415466, 0.44527941942214966],
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
batchmatmul_outputs = []
add_outputs = []
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
    # FC1 (lmu_kernel matches weights in tflite)
    x = tf.matmul(strided_slice_output, lmu_kernel)
    fc1_outputs.append(x)
    # ExpandDims
    x = tf.expand_dims(x, -1)
    # BatchMatMul with `B` matrix
    x = tf.matmul(x, batchmatmul_weights)
    batchmatmul_outputs.append(x)
    # Add
    x = tf.math.add(x, add_op_1_input)
    add_outputs.append(x)
    # Reshape
    x = tf.reshape(x, [-1, 4 * MEMORY_D])
    # FC (Dense layer) (dense_kernel matches weights in tflite)
    x = tf.matmul(x, dense_kernel)
    fc_dense_outputs.append(x)

# Run quantized forward pass
lmu_kernel_quant = get_fq_weights(lmu_kernel)
batchmatmul_weights_quant = get_fq_weights(batchmatmul_weights)
hidden_kernel_quant = get_fq_weights(hidden_kernel)
dense_kernel_quant = get_fq_weights(dense_kernel)
# Inputs to add ops are quantized asymmetrically
p = adjust_params(np.min(add_op_1_input), np.max(add_op_1_input))
add_op_1_weights_quant = tf.quantization.fake_quant_with_min_max_args(
    add_op_1_input, p[0], p[1]
)
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
    "batchmatmul_outputs",
    *calculate_scale_zp_from_min_max(np.min(batchmatmul_outputs), np.max(batchmatmul_outputs))
)
print(
    "add_outputs", *calculate_scale_zp_from_min_max(np.min(add_outputs), np.max(add_outputs))
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
    x = tf.matmul(strided_slice_output, lmu_kernel_quant)
    p = adjust_params(np.min(fc1_outputs), np.max(fc1_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    # ExpandDims
    x = tf.expand_dims(x, -1)
    # BatchMatMul with `B` matrix
    x = tf.matmul(x, batchmatmul_weights_quant)
    p = adjust_params(np.min(batchmatmul_outputs), np.max(batchmatmul_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    # Add
    x = tf.math.add(x, add_op_1_weights_quant)
    p = adjust_params(np.min(add_outputs), np.max(add_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    # Reshape
    x = tf.reshape(x, [-1, 4 * MEMORY_D])
    # FC2 (hidden_kernel matches weights in tflite)
    # FC (Dense layer) (dense_kernel matches weights in tflite)
    x = tf.matmul(x, dense_kernel_quant)
    p = adjust_params(np.min(fc_dense_outputs), np.max(fc_dense_outputs))
    x = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
    manual_outputs.append(x)

##################################################
# tflite computation
##################################################
# Define keras model with an LMU
inp = tf.keras.Input((TIMESTEPS, INPUT_D))
x = nengo_edge.layers.RNN(
    keras_lmu.LMUCell(
        memory_d=MEMORY_D,
        order=4,
        theta=5,
        hidden_cell=None,
        hidden_to_memory=False,
        memory_to_memory=False,
        input_to_hidden=False,
        kernel_initializer=tf.initializers.constant(lmu_kernel),
        recurrent_initializer=tf.initializers.constant(lmu_recurrent),
    )
)
x = x(inp, initial_state=[tf.ones([1, 4 * MEMORY_D])])
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
with open("saved_models/lmu.tflite", "wb") as f:
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
