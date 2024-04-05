"""Tests for `pypesto.sample` methods."""

import numpy as np
import pytest
from scipy.stats import kstest

import pypesto.optimize as optimize
from pypesto.variational import variational_fit

from ..sample.test_sample import (
    create_petab_problem,
    gaussian_mixture_problem,
    gaussian_problem,
    rosenbrock_problem,
)


def variational_petab_problem():
    # create problem
    problem = create_petab_problem()

    result = variational_fit(
        problem,
        n_iterations=100,
        method="advi",
        x0=np.array([3, -4]),
    )
    return result


@pytest.fixture(params=["gaussian", "gaussian_mixture", "rosenbrock"])
def problem(request):
    if request.param == "gaussian":
        return gaussian_problem()
    if request.param == "gaussian_mixture":
        return gaussian_mixture_problem()
    elif request.param == "rosenbrock":
        return rosenbrock_problem()


def test_pipeline(problem):
    """Check that a typical pipeline runs through."""
    # optimization
    optimizer = optimize.ScipyOptimizer(options={"maxiter": 10})
    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        optimizer=optimizer,
        progress_bar=False,
    )

    # sample
    result = variational_fit(
        problem=problem,
        n_iterations=100,
        n_samples=10,
        result=result,
    )


def test_ground_truth():
    """Test whether we actually retrieve correct distributions."""
    problem = gaussian_problem()

    result = optimize.minimize(
        problem,
        progress_bar=False,
    )

    result = variational_fit(
        problem,
        n_iterations=10000,
        n_samples=5000,
        result=result,
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0].flatten()

    # test against different distributions
    statistic, pval = kstest(samples, "norm")
    print(statistic, pval)
    assert statistic < 0.1

    statistic, pval = kstest(samples, "uniform")
    print(statistic, pval)
    assert statistic > 0.1
