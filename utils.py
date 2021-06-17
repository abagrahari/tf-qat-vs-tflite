import warnings

import numpy as np


def output_stats(x: np.ndarray, y: np.ndarray, test_name: str, model: str, tol: float):
    """Output summary statistics"""
    assert x.shape == y.shape

    outputs_close = np.allclose(x, y, rtol=0, atol=tol)
    status = "Passed" if outputs_close else "Failed"
    # Number of elements not matching
    num_mismatch = np.count_nonzero(~np.isclose(x, y, rtol=0, atol=tol))

    err = np.abs(x - y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        err_rel = err / np.abs(y)
        # Ignore "divide by zero" RuntimeWarning
    # Filter out nan and inf created by dividing by 0
    err_rel = err_rel[np.isfinite(err_rel)]

    print(f"--------------------- {test_name.upper()} ---------------------")
    print(f"Model: {model}; TestStatus: {status}; Tolerance: {tol}")
    print(f"Max Error: {np.max(err)}")
    print(f"Max Relative Error: {np.max(err_rel)}")
    print(f"Mean Error: {np.mean(err)}")
    print(f"Outputs not close: {num_mismatch/x.shape[0]*100}% of {x.shape[0]}")
    print()
