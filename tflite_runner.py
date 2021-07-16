import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def create_tflite_model(train_images, keras_model, model_path):
    if Path(model_path).exists():
        with open(model_path, "rb") as f:
            tflite_model = f.read()
        return tflite_model

    def representative_dataset():
        # Use the same inputs as what QAT model saw for calibration
        for data in (
            tf.data.Dataset.from_tensor_slices(train_images)
            .batch(1)
            .take(-1)  # Use all of dataset
        ):
            yield [tf.dtypes.cast(data, tf.float32)]

    # TF's QAT example uses Dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # For all INT8 conversion, we need some additional converter settings:
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8 for Coral
    converter.inference_output_type = tf.int8  # or tf.uint8 for Coral
    converter.representative_dataset = representative_dataset
    tflite_model = converter.convert()

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    with open(model_path, "wb") as f:
        f.write(tflite_model)
    return tflite_model


def get_interpreter(tflite_model):
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_preserve_all_tensors=True
        # experimental_preserve_all_tensors=True to be able to look at
        # intermediate outputs
    )
    interpreter.allocate_tensors()
    """Helper function to setup the tflite interpreter using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    return interpreter


def run_tflite_model(tflite_model, images_dataset):
    interpreter = get_interpreter(tflite_model)
    """Helper function to return outputs on supplied dataset using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Run predictions on every image in the "test" dataset.
    outputs = []
    for img in images_dataset:

        # Check if the input type is quantized, then rescale input data to uint8
        # as shown in TF's Post-Training Integer Quantization Example
        if input_details["dtype"] in [np.uint8, np.int8]:
            input_scale, input_zero_point = input_details["quantization"]
            img = img / input_scale + input_zero_point
            # The TensorFlow example does not have the np.round() op below.
            # However during testing, without it, values like `125.99998498`
            # are replaced with 125 instead of 126, since we would directly
            # cast to int8/uint8
            img = np.round(img)

        # Pre-processing: add batch dimension and convert to datatype to match with
        # the model's input data format.
        img = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], img)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and dequantize the output
        # based on TensorFlow's quantization params
        # We dequantize the outputs so we can directly compare the raw
        # outputs with the QAT model
        output = interpreter.get_tensor(output_details["index"])[0]
        if output_details["dtype"] in [np.uint8, np.int8]:
            output_scale, output_zero_point = output_details["quantization"]
            output = (output.astype(np.float32) - output_zero_point) * output_scale
        outputs.append(output)

    return np.array(outputs)


def collect_intermediate_outputs(tflite_model, images_dataset):
    interpreter = get_interpreter(tflite_model)
    """Helper function to return intermediate outputs on supplied dataset using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    input_details = interpreter.get_input_details()[0]
    tensor_details = interpreter.get_tensor_details()

    # Run predictions on every image in the "test" dataset.
    outputs = [[], [], [], [], []]

    for img in images_dataset:

        # Check if the input type is quantized, then rescale input data to uint8
        # as shown in TF's Post-Training Integer Quantization Example
        assert input_details["dtype"] in [np.uint8, np.int8]
        input_scale, input_zero_point = input_details["quantization"]
        img = img / input_scale + input_zero_point
        # The TensorFlow example does not have the np.round() op below.
        # However during testing, without it, values like `125.99998498`
        # are replaced with 125 instead of 126, since we would directly
        # cast to int8/uint8
        img = np.round(img)

        # Pre-processing: add batch dimension and convert to datatype to match with
        # the model's input data format.
        assert img.shape == (28, 28)
        img = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        assert img.shape == (1, 28, 28)
        interpreter.set_tensor(input_details["index"], img)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and dequantize the output
        # based on TensorFlow's quantization params
        # We dequantize the outputs so we can directly compare the raw
        # outputs with the QAT model
        for i in range(10, 15):
            layer_output = interpreter.get_tensor(i)[0]
            scale, zp = tensor_details[i]["quantization"]
            layer_output = (layer_output.astype(np.float32) - zp) * scale
            outputs[i - 10].append(layer_output)

    return [tf.convert_to_tensor(np.array(layer_outputs)) for layer_outputs in outputs]


def collect_intermediate_outputs_no_bias(tflite_model, images_dataset):
    interpreter = get_interpreter(tflite_model)
    """Helper function to return intermediate outputs on supplied dataset using the TF Lite model."""
    # https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
    input_details = interpreter.get_input_details()[0]
    tensor_details = interpreter.get_tensor_details()

    # Run predictions on every image in the "test" dataset.
    outputs = [[], [], [], [], []]

    for img in images_dataset:

        # Check if the input type is quantized, then rescale input data to uint8
        # as shown in TF's Post-Training Integer Quantization Example
        assert input_details["dtype"] in [np.uint8, np.int8]
        input_scale, input_zero_point = input_details["quantization"]
        img = img / input_scale + input_zero_point
        # The TensorFlow example does not have the np.round() op below.
        # However during testing, without it, values like `125.99998498`
        # are replaced with 125 instead of 126, since we would directly
        # cast to int8/uint8
        img = np.round(img)

        # Pre-processing: add batch dimension and convert to datatype to match with
        # the model's input data format.
        assert img.shape == (28, 28)
        img = np.expand_dims(img, axis=0).astype(input_details["dtype"])
        assert img.shape == (1, 28, 28)
        interpreter.set_tensor(input_details["index"], img)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and dequantize the output
        # based on TensorFlow's quantization params
        # We dequantize the outputs so we can directly compare the raw
        # outputs with the QAT model
        for i in range(6, 11):
            layer_output = interpreter.get_tensor(i)[0]
            scale, zp = tensor_details[i]["quantization"]
            layer_output = (layer_output.astype(np.float32) - zp) * scale
            outputs[i - 6].append(layer_output)

    return [tf.convert_to_tensor(np.array(layer_outputs)) for layer_outputs in outputs]


def evaluate_tflite_model(tflite_model, test_images, test_labels):
    """Helper function to evaluate the TF Lite model on the test dataset."""

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    outputs = run_tflite_model(tflite_model, test_images)
    for output in outputs:
        digit = np.argmax(output)
        prediction_digits.append(digit)

    # Compare prediction results with ground truth labels to calculate accuracy.
    prediction_digits = np.array(prediction_digits)
    accuracy = (prediction_digits == test_labels).mean()
    return accuracy
