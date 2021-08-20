# tf-qat-vs-tflite
Determining expected error between QAT and tflite interpreter for some simple models

Using the skeleton from [TensorFlow's QAT Example](https://www.tensorflow.org/model_optimization/guide/quantization/training_example)

## Environment setup:
- `conda env create --file environment.yml`
- `conda activate tf-exp`

## Running

To compare the QAT & custom model with the tflite models, we use a 2-step approach:
```bash
python main.py
python main.py --eval
```

or to iterate through the different seeds:

```bash
./run.sh
```

## Files:

| File Name                        | Goal                                                                                                                                                           | Start here? |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| `compare_dense.py`               | Minimal version of `main.py`. Comparing use of the CustomLayer with tflite parameters from tflite model, vs. the tflite model                                  |             |
| `compare_manual_fq_w_qat.py`     | Example of manual computations needed to match a QAT model, using dense model                                                                                  | Yes         |
| `compare_manual_lmu_w_tflite.py` | Trying to perform manual computations for lmu-based model to match a quantized tflite model                                                                    | Yes         |
| `compare_manual_w_tflite.py`     | Example of manual computations needed to match a quantized tflite model, using dense model                                                                     | Yes         |
| `compare_params.py`              | Determining if QAT and tflite quantization parameters are the same                                                                                             |             |
| `custom_layers.py`               | Methods to adjust quantization parameters like TFLite does, and a CustomLayer that uses quant parameters from tflite model                                     | Yes         |
| `environment.yml`                | Environment file                                                                                                                                               |             |
| `github_issue.ipynb`             | File linked in Github issue to demo QAT vs tflite quantization. See https://git.io/J0hWR instead                                                               |             |
| `github_issue.py`                | See above                                                                                                                                                      |             |
| `github_issue_compounding.ipynb` | See above                                                                                                                                                      |             |
| `github_issue_compounding.py`    | See above                                                                                                                                                      |             |
| `main.py`                        | Comparing QAT model, use of the CustomLayer with tflite parameters from tflite model, and the tflite model                                                     |             |
| `pyproject.toml`                 | Formatting file                                                                                                                                                |             |
| `run.sh`                         | Short script to run `main.py` with different seeds, and collect results                                                                                        |             |
| `test_dequant.py`                | Scratch program made to resolve an overflow issue. Can ignore.                                                                                                 |             |
| `tflite_runner.py`               | Functions to help create and run a tflite model                                                                                                                |             |
| `training_example.ipynb`         | Modified version of TensorFlow's original QAT training tutorial from [here](https://www.tensorflow.org/model_optimization/guide/quantization/training_example) |             |
| `utils.py`                       | Helper functions to print summary statistics between outputs                                                                                                   |             |
| `verify_tflite.py`               | Scratch program to try extract intermediate outputs from tflite model                                                                                          |             |

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
