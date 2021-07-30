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

default_8bit_quantize_registry.quantizers.MovingAverageQuantizer = (
    tfmot.quantization.keras.quantizers.AllValuesQuantizer
)


def calculate_min_max_for_fake_quant(
    values,
):
    """Calculate min/max quantization parameters for use in tf.quantization.fake_quant_with_min_max_args.
    Calculates as per AllValuesQuantize implementation https://git.io/JBVeP
    Args:
      values: a tensor containing values to be quantized.
    Returns:
      min/max parameters for use in fake_quantize_with... as per QAT.
    """
    min = np.min(values)
    max = np.max(values)
    # TFLite requires that 0.0 if always in the [min; max] range.
    min = tf.math.minimum(min, 0.0)
    max = tf.math.maximum(max, 0.0)
    return min.numpy(), max.numpy()


rng = np.random.RandomState(0)
x = rng.uniform(0, 1, size=(32, 10))
w = rng.uniform(-1, 1, size=(10, 10))
N_LAYERS = 5

##################################################
# Manual computation - QAT
##################################################
# QAT performs fake_quantizing during the forward pass
# and fake_quant_with_min_max internally nudges the parameters
w_quant = tf.quantization.fake_quant_with_min_max_args(
    w, min(np.min(w), -np.max(w)), max(np.max(w), -np.min(w)), narrow_range=True
)
manual_output = x
for i in range(N_LAYERS):
    manual_output = tf.quantization.fake_quant_with_min_max_args(
        manual_output, *calculate_min_max_for_fake_quant(manual_output)
    )
    manual_output = tf.matmul(manual_output, w_quant)
manual_output = tf.quantization.fake_quant_with_min_max_args(
    manual_output, *calculate_min_max_for_fake_quant(manual_output)
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
