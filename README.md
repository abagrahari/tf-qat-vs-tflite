# tf-qat-vs-tflite
Determining expected error between QAT and tflite interpreter for some simple models

Using the skeleton from [TensorFlow's QAT Example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

## Environment setup:
- `conda env create --file environment.yml`
- `conda activate tf-exp`
- `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- `jupyter-lab`

## TF's Documentation on running a model with tflite
- https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
- https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=tfer6hI8ljEh
- Converting to tflite
    - https://www.tensorflow.org/lite/performance/post_training_quantization#dynamic_range_quantization
