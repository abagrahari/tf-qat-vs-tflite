import csv
import warnings
from pathlib import Path

import numpy as np
from tensorflow import keras


def load_mnist():
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the images so that each pixel value is between 0 to 1.
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels), (test_images, test_labels)


def output_stats(
    x: np.ndarray,
    y: np.ndarray,
    test_name: str,
    model: str,
    tol: float,
    seed: int,
):
    """Output summary statistics"""
    assert x.shape == y.shape

    outputs_close = np.allclose(x, y, rtol=0, atol=tol)
    status = "Passed" if outputs_close else "Failed"
    # Number of elements not matching
    num_mismatch = np.count_nonzero(~np.isclose(x, y, rtol=0, atol=tol))

    # Error is defined as the difference between the two outputs
    err = np.abs(x - y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        err_rel = err / np.abs(y)
        # Ignore "divide by zero" RuntimeWarning
    # Filter out nan and inf created by dividing by 0
    err_rel = err_rel[np.isfinite(err_rel)]

    print(f"--------------------- {test_name.upper()} ---------------------")
    print(f"Model: {model}; TestStatus: {status}; Tolerance: {tol}; Seed: {seed}")
    print(f"Max Error: {np.max(err)}")
    print(f"Max Relative Error: {np.max(err_rel)}")
    print(f"Mean Error: {np.mean(err)}")
    print(f"Outputs not close: {num_mismatch/x.shape[0]*100}% of {x.shape[0]}")
    print()

    filename = "summary.csv"
    my_file = Path(filename)
    if not my_file.exists():
        with open(filename, mode="w") as summary_file:
            summary_writer = csv.writer(
                summary_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            summary_writer.writerow(
                [
                    "Model",
                    "Seed",
                    "Test Status",
                    "Max Error",
                    "Max Relative Error",
                    "Mean Error",
                    "Percent Mismatched",
                ]
            )
    with open(filename, mode="a") as summary_file:
        summary_writer = csv.writer(
            summary_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        summary_writer.writerow(
            [
                model,
                seed,
                status,
                np.max(err),
                np.max(err_rel),
                np.mean(err),
                num_mismatch / x.shape[0] * 100,
            ]
        )
