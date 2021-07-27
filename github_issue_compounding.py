import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

# manual computation
rng = np.random.RandomState(0)
x = rng.uniform(0, 1, size=(32, 10))
w = rng.uniform(-1, 1, size=(10, 10))
N_LAYERS = 5

# As per TfLite's QuantizeModel https://git.io/J4hxt, it seems that a full fp32 forward pass is done
# after which, quantization parameters are independantly calculated. Then, the model is 'quantized'

# Run fp32 forward pass
manual_output = x
fp32_outputs = []
for _ in range(N_LAYERS):
    manual_output = tf.matmul(manual_output, w)
    fp32_outputs.append(manual_output)

# Run quantized pass after 'computing' quantization parameters
w_quant = tf.quantization.fake_quant_with_min_max_args(
    w, min(np.min(w), -np.max(w)), max(np.max(w), -np.min(w)), narrow_range=True
)
manual_output = tf.quantization.fake_quant_with_min_max_args(x, np.min(x), np.max(x))
for i in range(N_LAYERS):
    manual_output = tf.matmul(manual_output, w_quant)
    manual_output = tf.quantization.fake_quant_with_min_max_args(
        manual_output,
        np.min(fp32_outputs[i]),
        np.max(fp32_outputs[i])
        # Use min/max of fp32 forward pass for quantization parameters
    )

# tflite computation
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

# compare outputs
outputs_close = np.allclose(manual_output, tflite_output, rtol=0, atol=1e-2)
# Number of elements not within the tolerance
num_mismatch = np.count_nonzero(~np.isclose(manual_output, tflite_output, rtol=0, atol=1e-2))
err = np.abs(manual_output - tflite_output)
with warnings.catch_warnings():
    # Ignore "divide by zero" RuntimeWarning
    warnings.simplefilter("ignore")
    err_rel = err / np.abs(tflite_output)
# Filter out nan and inf created by dividing by 0
err_rel = err_rel[np.isfinite(err_rel)]
print(f"--------------------- Manual vs TFLite ---------------------")
print(f"Max Error: {np.max(err)}")
print(f"Max Relative Error: {np.max(err_rel)}")
print(f"Mean Error: {np.mean(err)}")
print(f"Number of outputs outside tolerance: {num_mismatch/x.size*100}% of {x.size}")
