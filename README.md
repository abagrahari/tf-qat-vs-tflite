# tf-qat-vs-tflite
Determining expected error between QAT and tflite interpreter for some simple models

Using the skeleton from [TensorFlow's QAT Example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

## Environment setup:
- `conda env create --file environment.yml`
- `conda activate tf-exp`
- `jupyter labextension install @jupyter-widgets/jupyterlab-manager`
- `jupyter-lab`

## Running

```bash
python main.py --model dense4 --quantize
```

or to iterate through the options

```bash
./run.sh
```

## TF's Documentation on running a model with tflite
- Running a tflite model
    - https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python
- https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/digit_classifier/ml/step2_train_ml_model.ipynb#scrollTo=ahTP3T60nYJb
- Converting to tflite, with quantization
    - https://www.tensorflow.org/lite/performance/post_training_quantization
    - https://www.tensorflow.org/lite/performance/post_training_integer_quant
    - https://stackoverflow.com/questions/62015923/why-the-accuracy-of-tf-lite-is-not-correct-after-quantization

## Results

Sample output
```
$ python main.py --model dense4 --quantize

1688/1688 [==============================] - 3s 1ms/step - loss: 0.5426 - accuracy: 0.8385 - val_loss: 0.3059 - val_accuracy: 0.9142
2/2 [==============================] - 1s 148ms/step - loss: 0.6230 - accuracy: 0.8311 - val_loss: 0.5737 - val_accuracy: 0.8600

Baseline test accuracy: 0.894599974155426
QAT test accuracy: 0.8363999724388123
TFLite test_accuracy: 0.8365

--------------------- BASELINE VS QAT ---------------------
Model: dense4; TestStatus: Failed; Tolerance: 0.01; Seed: 3
Max Error: 22.29353904724121
Max Relative Error: 136.0284881591797
Mean Error: 1.2580227851867676
Outputs not close: 99.279% of 100000

--------------------- BASELINE VS TFLITE ---------------------
Model: dense4; TestStatus: Failed; Tolerance: 0.01; Seed: 3
Max Error: 22.29353904724121
Max Relative Error: 136.0279083251953
Mean Error: 1.2580808401107788
Outputs not close: 99.28200000000001% of 100000

--------------------- QAT VS TFLITE ---------------------
Model: dense4; TestStatus: Failed; Tolerance: 0.01; Seed: 3
Max Error: 0.1887655258178711
Max Relative Error: 1.9999957084655762
Mean Error: 0.0006519349408335984
Outputs not close: 1.3% of 100000
```