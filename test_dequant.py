# Short scratch program to determine differences between
# similar looking dequantization formulas/methods

import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

int8_val = tf.constant([1, 3, 54, 3, -8], dtype=tf.int8)
output_scale: float = 0.003921568859368563
output_zp: int = -128


def compare(y, z):
    print("y: ", y)
    print("z: ", z)
    print("y == z?: ", np.allclose(y, z, rtol=0, atol=1e-4))


y = (tf.cast(int8_val, tf.float32) - output_zp) * output_scale
# using next line instead of above line causes drop in accuracy
# There seems to be a difference between the two methods
z = tf.cast((int8_val - output_zp), tf.float32) * output_scale
print("\nWith int8 input and output scale")
compare(y, z)

# Solution - it is due to integer overflow!!
y = tf.cast(int8_val, tf.float32) + 128  # this is the correct method. (Cast first)
z = tf.cast((int8_val + 128), tf.float32)
# Second method has int overflow since cast is performed after addition
# e.g. 3 + 128 > 128 ==> overflows to -125
print("\nWith int8 input")
compare(y, z)
