import os
import tempfile

import numpy as np
import pytest
import scipy.optimize as so

import pypesto
import pypesto.optimize as optimize
from pypesto.store import write_result


@pytest.fixture
def hdf5_file():
    """Generate a temporary hdf5 file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        file = os.path.join(tmp_dir, "file.hdf5")
        yield file


@pytest.fixture
def optimized_result_with_history(hdf5_file):
    """Create and optimize a problem with history, write to hdf5 file.

    Returns the result object after optimization and writing to file.
    This fixture is used to test history loading in both lazy and non-lazy modes.
    """
    # Create objective
    objective = pypesto.Objective(
        fun=so.rosen, grad=so.rosen_der, hess=so.rosen_hess
    )

    # Create problem
    dim_full = 5
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    # Create optimizer
    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})

    # Create history options
    history_options = pypesto.HistoryOptions(
        trace_record=True, storage_file=hdf5_file
    )

    # Optimize with history
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=3,
        history_options=history_options,
        progress_bar=False,
    )

    # Write result to file
    write_result(result=result, filename=hdf5_file, overwrite=True)

    return result
