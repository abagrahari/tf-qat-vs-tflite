import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.default_8bit import (
    default_8bit_quantize_registry,
)
import utils

# MonkeyPatch to use AllValuesQuantizer
default_8bit_quantize_registry.quantizers.MovingAverageQuantizer = (
    tfmot.quantization.keras.quantizers.AllValuesQuantizer
)


# manual computation
rng = np.random.RandomState(0)
x = rng.uniform(0, 1, size=(32, 10))
w = rng.uniform(-1, 1, size=(10, 10))
N_LAYERS = 5

w_quant = tf.quantization.fake_quant_with_min_max_args(
    w, min(np.min(w), -np.max(w)), max(np.max(w), -np.min(w)), narrow_range=True
)
manual_output = x
for i in range(N_LAYERS):
    manual_output = tf.quantization.fake_quant_with_min_max_args(
        manual_output, np.min(manual_output), np.max(manual_output)
    )
    manual_output = tf.matmul(manual_output, w_quant)
manual_output = tf.quantization.fake_quant_with_min_max_args(
    manual_output, np.min(manual_output), np.max(manual_output)
)

# QAT computation
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(
            10, use_bias=False, kernel_initializer=tf.initializers.constant(w)
        )
        for _ in range(N_LAYERS)
    ]
)
model.build(x.shape)
qat_model = tfmot.quantization.keras.quantize_model(model)
# Calibrate QAT model
qat_model(x, training=True)

# Compare outputs
qat_output = qat_model(x)
manual_output = np.array(manual_output).flatten()
qat_output = np.array(qat_output).flatten()
utils.output_stats(manual_output, qat_output, "Manual w/ FakeQuant vs QAT", 1e-2, 0)
