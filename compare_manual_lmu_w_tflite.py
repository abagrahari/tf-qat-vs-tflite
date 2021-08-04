import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import utils
from custom_layers import calculate_min_max_from_tflite, calculate_scale_zp_from_min_max

import keras_lmu


tf.random.set_seed(3)
rng = np.random.RandomState(3)
w = rng.uniform(-1, 1, size=(10, 10))


def adjust_params(min, max):
    """Adjust quantization parameters as tflite does"""
    return [
        t.numpy()
        for t in calculate_min_max_from_tflite(*calculate_scale_zp_from_min_max(min, max))
    ]


##################################################
# Manual computation - tflite
##################################################
# TODO: adapt from compare_manual_w_tflite.py
##################################################
# tflite computation
##################################################
# Define keras model with an LMU
INPUT_D = 4
TIMESTEPS = 1

inputs = rng.uniform(-0.5, 0.5, size=(320, TIMESTEPS, INPUT_D))
# 320 different inputs/sequences, of TIMESTEPS timesteps, and INPUT_D dimensions/features in each input

model = tf.keras.Sequential(
    [
        tf.keras.layers.InputLayer((TIMESTEPS, INPUT_D)),
        keras_lmu.LMU(
            memory_d=1,
            order=4,
            theta=5,
            hidden_cell=tf.keras.layers.SimpleRNNCell(
                units=10,
                activation="relu",
            ),
            hidden_to_memory=True,
            memory_to_memory=True,
            input_to_hidden=True,
            # TODO: specify kernel intializer and recurrent initializer
        ),
        tf.keras.layers.Dense(
            10,
            use_bias=False,
            # TODO: try with bias
            kernel_initializer=tf.initializers.constant(w),
        ),
    ]
)

model.build(inputs.shape)
print(inputs.shape)
model.summary()
manual_output = model(inputs)
print(manual_output)
# TODO is initial_state needed if predicting using the lmu model?


def representative_dataset():
    for i in range(inputs.shape[0]):
        yield [inputs[[i]].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
# converter.representative_dataset = representative_dataset

interpreter = tf.lite.Interpreter(model_content=converter.convert())
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


inputs_quant = inputs.astype(np.float32)
# input_scale, input_zero_point = input_details["quantization"]
# inputs_quant = np.round(inputs / input_scale + input_zero_point).astype(np.uint8)
interpreter.set_tensor(input_details["index"], inputs_quant)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details["index"])
# output_scale, output_zero_point = output_details["quantization"]
# tflite_output = (tflite_output.astype(np.float32) - output_zero_point) * output_scale

# Compare outputs
manual_output = np.array(manual_output).flatten()
tflite_output = np.array(tflite_output).flatten()
utils.output_stats(manual_output, tflite_output, "Manual LMU vs tflite", 1e-2, 0)
