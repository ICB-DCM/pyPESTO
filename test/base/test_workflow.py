"""Test a basic workflow, with minimum dependencies.

These tests are not for correctness, but for basic functionality.
"""

from functools import wraps

import matplotlib.pyplot as plt

import pypesto
import pypesto.optimize as optimize
import pypesto.profile as profile
import pypesto.sample as sample
import pypesto.visualize as visualize

from ..util import CRProblem


def close_fig(fun):
    """Close figure."""

    @wraps(fun)
    def wrapped_fun(*args, **kwargs):
        ret = fun(*args, **kwargs)
        plt.close("all")
        return ret

    return wrapped_fun


def test_objective():
    """Test a simple objective function."""
    crproblem = CRProblem()
    obj = crproblem.get_objective()
    p = crproblem.p_true

    assert obj(p) == crproblem.get_fnllh()(p)
    assert obj(p, sensi_orders=(0,)) == crproblem.get_fnllh()(p)
    assert (obj(p, sensi_orders=(1,)) == crproblem.get_fsnllh()(p)).all()
    assert (obj(p, sensi_orders=(2,)) == crproblem.get_fs2nllh()(p)).all()
    fval, grad = obj(p, sensi_orders=(0, 1))
    assert fval == crproblem.get_fnllh()(p)
    assert (grad == crproblem.get_fsnllh()(p)).all()


@close_fig
def test_optimize():
    """Test a simple multi-start optimization."""
    crproblem = CRProblem()
    problem = pypesto.Problem(
        objective=crproblem.get_objective(),
        lb=crproblem.lb,
        ub=crproblem.ub,
    )
    optimizer = optimize.ScipyOptimizer()
    n_start = 20
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_start,
    )

    # check basic sanity
    assert len(result.optimize_result.list) == n_start
    assert len(result.optimize_result.fval) == n_start
    assert len(result.optimize_result.x) == n_start

    # check that the results are sorted
    fvals = result.optimize_result.fval
    assert fvals == sorted(fvals)

    # check that optimization was successful
    assert fvals[0] < crproblem.get_fnllh()(crproblem.p_true)

    # visualize the results
    visualize.waterfall(result)


@close_fig
def test_profile():
    """Test a simple profile calculation."""
    crproblem = CRProblem()
    problem = pypesto.Problem(
        objective=crproblem.get_objective(),
        lb=crproblem.lb,
        ub=crproblem.ub,
    )
    optimizer = optimize.ScipyOptimizer()
    n_starts = 5
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_starts,
    )
    profile_result = profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=optimizer,
        profile_index=[0],
    )

    # check basic sanity
    assert len(profile_result.profile_result.list) == 1

    # visualize the results
    visualize.profiles(profile_result)


@close_fig
def test_sample():
    """Test a simple sampling."""
    crproblem = CRProblem()
    problem = pypesto.Problem(
        objective=crproblem.get_objective(),
        lb=crproblem.lb,
        ub=crproblem.ub,
    )
    optimizer = optimize.ScipyOptimizer()
    n_start = 5
    result = optimize.minimize(
        problem=problem,
        optimizer=optimizer,
        n_starts=n_start,
    )
    sampler = sample.AdaptiveMetropolisSampler()
    n_samples = 500
    sample_result = sample.sample(
        problem=problem,
        result=result,
        sampler=sampler,
        n_samples=n_samples,
    )

    # check basic sanity
    assert sample_result.sample_result.trace_x.shape == (
        1,
        n_samples + 1,
        len(crproblem.p_true),
    )
    sample.geweke_test(sample_result)
    # visualize the results
    visualize.sampling_1d_marginals(sample_result)
