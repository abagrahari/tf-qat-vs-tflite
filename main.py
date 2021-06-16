# End-to-end example for quantization aware training.

# In this, we:

# Train a tf.keras model for MNIST from scratch.
# Fine tune the model by applying the quantization aware training API, see the accuracy, and export a quantization aware model.
# Use the model to create an actually quantized model for the TFLite backend.
# See the persistence of accuracy in TFLite and a 4x smaller model.

import os
import tempfile

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = 2
from tensorflow import keras

# Train a model for MNIST without quantization aware training

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture.
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape=(28, 28)),
        keras.layers.Reshape(target_shape=(28, 28, 1)),
        keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(10),
    ]
)

# Train the digit classification model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.fit(
    train_images,
    train_labels,
    epochs=1,
    validation_split=0.1,
)


# Clone and fine-tune pre-trained model with quantization aware training
# Define the model
# You will apply quantization aware training to the whole model and see this in the model summary. All layers are now prefixed by "quant".

# Note that the resulting model is quantization aware but not quantized (e.g. the weights are float32 instead of int8). The sections after show how to create a quantized model from the quantization aware one.

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

q_aware_model.summary()

# Train and evaluate the model against baseline
# To demonstrate fine tuning after training the model for just an epoch, fine tune with quantization aware training on a subset of the training data.

train_images_subset = train_images[0:1000]  # out of 60000
train_labels_subset = train_labels[0:1000]

q_aware_model.fit(
    train_images_subset,
    train_labels_subset,
    batch_size=500,
    epochs=1,
    validation_split=0.1,
)

# For this example, there is minimal to no loss in test accuracy after quantization aware training, compared to the baseline.
_, baseline_model_accuracy = model.evaluate(test_images, test_labels, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(test_images, test_labels, verbose=0)

print("Baseline test accuracy:", baseline_model_accuracy)
print("Quant test accuracy:", q_aware_model_accuracy)

# Create quantized model for TFLite backend
# After this, you have an actually quantized model with int8 weights and uint8 activations.

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

# See persistence of accuracy from TF to TFLite
# Define a helper function to evaluate the TF Lite model on the test dataset.

from pprint import pprint


def evaluate_model(interpreter):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    pprint(interpreter.get_input_details())
    pprint(interpreter.get_output_details())
    print()

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    for i, test_image in enumerate(test_images):
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy


# You evaluate the quantized model and see that the accuracy from TensorFlow persists to the TFLite backend.
interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print("Quant TF test accuracy:", q_aware_model_accuracy)
print("Quant TFLite test_accuracy:", test_accuracy)
# See 4x smaller model from quantization
# You create a float TFLite model and then see that the quantized TFLite model is 4x smaller.

# Create float TFLite model.
float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

# Measure sizes of models.
_, float_file = tempfile.mkstemp(".tflite")
_, quant_file = tempfile.mkstemp(".tflite")

with open(quant_file, "wb") as f:
    f.write(quantized_tflite_model)

with open(float_file, "wb") as f:
    f.write(float_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2 ** 20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2 ** 20))

# Run on QAT model
qat_output = q_aware_model.predict(test_images)
# Run on simulated device
# host_sim_output = host_sim.run(inputs)

qat_output = qat_output.flatten()
# host_sim_output = host_sim_output.flatten()
print(qat_output)

# Conclusion
# In this tutorial, you saw how to create quantization aware models with the TensorFlow Model Optimization Toolkit API and then quantized models for the TFLite backend.

# You saw a 4x model size compression benefit for a model for MNIST, with minimal accuracy difference.
