import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import utils
from custom_layers import calculate_min_max_from_tflite, calculate_scale_zp_from_min_max

rng = np.random.RandomState(0)
x = rng.uniform(0, 1, size=(32, 10))
w = rng.uniform(-1, 1, size=(10, 10))
N_LAYERS = 5


def adjust_params(min, max):
    """Adjust quantization parameters as tflite does"""
    return [
        t.numpy()
        for t in calculate_min_max_from_tflite(*calculate_scale_zp_from_min_max(min, max))
    ]


##################################################
# Manual computation - tflite
##################################################
# As per TfLite's QuantizeModel https://git.io/J4hxt, it seems that a full fp32 forward pass is done
# after which, quantization parameters are independantly calculated. Then, the model is 'quantized'

# Run fp32 forward pass
manual_output = x
fp32_outputs = []
for _ in range(N_LAYERS):
    manual_output = tf.matmul(manual_output, w)
    fp32_outputs.append(manual_output)

# Run quantized pass
w_quant = tf.quantization.fake_quant_with_min_max_args(
    w, min(np.min(w), -np.max(w)), max(np.max(w), -np.min(w)), narrow_range=True
)
# For the manual computation - we look at the min/max of the fp32 forward pass outputs.
# Then, we convert this min/max to tflite-nudged scale/zp, and then convert back to min/max
# for use in fake_quant_with_min_max.
p = adjust_params(np.min(x), np.max(x))
manual_output = tf.quantization.fake_quant_with_min_max_args(x, p[0], p[1])
for i in range(N_LAYERS):
    manual_output = tf.matmul(manual_output, w_quant)
    p = adjust_params(np.min(fp32_outputs[i]), np.max(fp32_outputs[i]))
    manual_output = tf.quantization.fake_quant_with_min_max_args(manual_output, p[0], p[1])


##################################################
# tflite computation
##################################################
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            10, use_bias=False, kernel_initializer=tf.initializers.constant(w)
        )
        for _ in range(N_LAYERS)
    ]
)
model.build(x.shape)


def representative_dataset():
    for i in range(x.shape[0]):
        yield [x[[i]].astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.representative_dataset = representative_dataset

interpreter = tf.lite.Interpreter(model_content=converter.convert())
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

input_scale, input_zero_point = input_details["quantization"]
x_quant = np.round(x / input_scale + input_zero_point).astype(np.uint8)
interpreter.set_tensor(input_details["index"], x_quant)
interpreter.invoke()
tflite_output = interpreter.get_tensor(output_details["index"])
output_scale, output_zero_point = output_details["quantization"]
tflite_output = (tflite_output.astype(np.float32) - output_zero_point) * output_scale

# Compare outputs
manual_output = np.array(manual_output).flatten()
tflite_output = np.array(tflite_output).flatten()
utils.output_stats(manual_output, tflite_output, "Manual vs tflite", 1e-2, 0)
