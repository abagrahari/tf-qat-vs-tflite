import warnings

import numpy as np


def output_stats(x, y, model: str, test_status: bool, tol: float):
    """Output summary statistics"""
    err = np.abs(x - y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        err_rel = err / np.abs(y)
        # Ignore "divide by zero" RuntimeWarning
    # Filter out nan and inf created by dividing by 0
    err_rel = err_rel[np.isfinite(err_rel)]

    status = "Passed" if test_status else "Failed"
    print(f"--------------------- RESULTS ---------------------")
    print(f"Model: {model}; TestStatus: {status}; Tolerance: {tol}")
    print(f"Max Error: {np.max(err)}")
    print(f"Max Relative Error: {np.max(err_rel)}")
    print(f"Mean Error: {np.mean(err)}")
    print("-------------------------------------------------")
