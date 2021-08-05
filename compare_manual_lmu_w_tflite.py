import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras_lmu
import nengo_edge
import numpy as np
import tensorflow as tf

import utils

tf.random.set_seed(3)
rng = np.random.RandomState(3)
w = rng.uniform(-1, 1, size=(10, 10))

##################################################
# tflite computation
##################################################
# Define keras model with an LMU
INPUT_D = 4
TIMESTEPS = 1

inputs = rng.uniform(-0.5, 0.5, size=(320, TIMESTEPS, INPUT_D))
# 320 different inputs/sequences, of TIMESTEPS timesteps, and INPUT_D dimensions/features in each input

inp = tf.keras.Input((TIMESTEPS, INPUT_D))
rnn = nengo_edge.layers.RNN(
    keras_lmu.LMUCell(
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
    )
)
out = rnn(inp, initial_state=[tf.zeros((1, 10)), tf.zeros((1, 4))])
model = tf.keras.Model(inp, out)

model_output = model.predict(inputs, batch_size=1)


def representative_dataset():
    for i in range(inputs.shape[0]):
        yield [inputs[[i]].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset

tflite_model = converter.convert()
print("---------Converted to tflite!---------")

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("---------Allocated tensors---------")

tflite_output = []
for input in inputs:
    input = np.expand_dims(input, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    tflite_output.append(output)
tflite_output = np.array(tflite_output)

# Compare outputs
model_output = np.array(model_output).flatten()
tflite_output = np.array(tflite_output).flatten()
utils.output_stats(model_output, tflite_output, "Manual LMU vs tflite", 1e-2, 0)
