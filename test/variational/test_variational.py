"""Tests for `pypesto.sample` methods."""

from scipy.stats import kstest

import pypesto.optimize as optimize
from pypesto.variational import variational_fit

from ..sample.test_sample import problem  # noqa: F401, fixture from sampling
from ..sample.util import STATISTIC_TOL, gaussian_problem


def test_pipeline(problem):  # noqa: F811
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
    variational_fit(
        problem=problem,
        n_iterations=100,
        n_samples=10,
        result=result,
    )


def test_ground_truth():
    """Test whether we actually retrieve correct distributions."""
    problem_gaussian = gaussian_problem()

    result = optimize.minimize(
        problem_gaussian,
        progress_bar=False,
    )

    result = variational_fit(
        problem_gaussian,
        n_iterations=10000,
        n_samples=5000,
        result=result,
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0].flatten()

    # test against different distributions
    statistic, pval = kstest(samples, "norm")
    print(statistic, pval)
    assert statistic < STATISTIC_TOL

    statistic, pval = kstest(samples, "uniform")
    print(statistic, pval)
    assert statistic > STATISTIC_TOL
