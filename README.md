# tf-qat-vs-tflite
Determining expected error between QAT and tflite interpreter for some simple models

Using the skeleton from [TensorFlow's QAT Example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

## Environment setup:
- `conda env create --file environment.yml`
- `conda activate tf-exp`
- `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- `jupyter-lab`

## Running

To compare the QAT model with the tflite model, we use a 2-step approach:
```bash
python main.py --model dense4
python main.py --model dense4 --eval
```

or to iterate through the options (seeds, model types, etc.):

```bash
./run.sh
```

To compare the custom dense layers with the tflite model:
```
python compare_dense.py
```

## TF's Documentation on running a model with tflite
- Running a tflite model
    - https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
- https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=ahTP3T60nYJb
- Converting to tflite, with quantization
    - https://www.tensorflow.org/lite/performance/post_training_quantization
    - https://www.tensorflow.org/lite/performance/post_training_integer_quant
    - https://stackoverflow.com/questions/62015923/why-the-accuracy-of-tf-lite-is-not-correct-after-quantization

## Saving weights of a model
- https://www.tensorflow.org/guide/keras/save_and_serialize#apis_for_saving_weights_to_disk_loading_them_back

## Recreating a tflite model in tensorflow
- TFLite quantizes values in [this way](https://www.tensorflow.org/lite/performance/quantization_spec#specification_summary):
    - `real_value = (int8_value - zero_point) * scale`
