import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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
    x: np.ndarray, y: np.ndarray, test_name: str, tol: float, seed: int, ax=None
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

    if ax is not None:

        hist, bins = np.histogram(err_rel, bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align="center", width=width)
        ax.set_xlabel("Error amount")
        ax.set_ylabel("Number of outupts")
        ax.set_title(f"{test_name}")
        if "2" in test_name:
            plt.figure()
            plt.bar(center, hist, align="center", width=width)
            plt.xlabel("Relative Error")
            plt.ylabel("Number of outupts")
            plt.title(f"{test_name}")

    print(f"--------------------- {test_name.upper()} ---------------------")
    print(f"TestStatus: {status}; Tolerance: {tol}; Seed: {seed}")
    print(f"Max Error: {np.max(err)}")
    print(f"Max Relative Error: {np.max(err_rel)}")
    print(f"Mean Error: {np.mean(err)}")
    print(f"Outputs not close: {num_mismatch/x.shape[0]*100}% of {x.shape[0]}")

    filename = "summary.csv"
    my_file = Path(filename)
    if not my_file.exists():
        with open(filename, mode="w") as summary_file:
            summary_writer = csv.writer(
                summary_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            summary_writer.writerow(
                [
                    "Test",
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
                test_name,
                seed,
                status,
                np.max(err),
                np.max(err_rel),
                np.mean(err),
                num_mismatch / x.shape[0] * 100,
            ]
        )


def get_max_rel_err(x: np.ndarray, y: np.ndarray, tol: float) -> float:
    """Get max relative error"""
    assert x.shape == y.shape
    # Error is defined as the difference between the two outputs
    err = np.abs(x - y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        err_rel = err / np.abs(y)
        # Ignore "divide by zero" RuntimeWarning
    # Filter out nan and inf created by dividing by 0
    err_rel = err_rel[np.isfinite(err_rel)]
    return np.max(err_rel)
