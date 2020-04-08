"""
This is for testing optimization of the pypesto.Objective.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm, kstest
import scipy.optimize as so
import matplotlib.pyplot as plt
import pytest

import pypesto


def gaussian_llh(x):
    return float(norm.logpdf(x))


def gaussian_problem():
    def nllh(x):
        return - gaussian_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(objective=objective, lb=[-10], ub=[10])
    return problem


def gaussian_mixture_llh(x):
    return np.log(
        0.3 * multivariate_normal.pdf(x, mean=-1.5, cov=0.1)
        + 0.7 * multivariate_normal.pdf(x, mean=2.5, cov=0.2))


def gaussian_mixture_problem():
    """Problem based on a mixture of gaussians."""
    def nllh(x):
        return - gaussian_mixture_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(objective=objective, lb=[-10], ub=[10],
                              x_names=['x'])
    return problem


def rosenbrock_problem():
    """Problem based on rosenbrock objective."""
    objective = pypesto.Objective(fun=so.rosen)

    dim_full = 2
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)
    return problem


@pytest.fixture(params=['Metropolis',
                        'AdaptiveMetropolis',
                        'ParallelTempering',
                        'AdaptiveParallelTempering'])
def sampler(request):
    if request.param == 'Metropolis':
        return pypesto.MetropolisSampler()
    elif request.param == 'AdaptiveMetropolis':
        return pypesto.AdaptiveMetropolisSampler()
    elif request.param == 'ParallelTempering':
        return pypesto.ParallelTemperingSampler(
            internal_sampler=pypesto.MetropolisSampler(),
            betas=[1, 1e-2, 1e-4])
    elif request.param == 'AdaptiveParallelTempering':
        return pypesto.AdaptiveParallelTemperingSampler(
            internal_sampler=pypesto.AdaptiveMetropolisSampler(),
            n_chains=5)


@pytest.fixture(params=['gaussian', 'gaussian_mixture', 'rosenbrock'])
def problem(request):
    if request.param == 'gaussian':
        return gaussian_problem()
    if request.param == 'gaussian_mixture':
        return gaussian_mixture_problem()
    elif request.param == 'rosenbrock':
        return rosenbrock_problem()


def test_pipeline(sampler, problem):
    """Check that a typical pipeline runs through."""
    # optimization
    optimizer = pypesto.ScipyOptimizer(options={'maxiter': 10})
    result = pypesto.minimize(problem, n_starts=3, optimizer=optimizer)

    # sampling
    result = pypesto.sample(
        problem, sampler=sampler, n_samples=20, result=result)

    # some plot
    pypesto.visualize.sampling_1d_marginals(result)
    plt.close()


def test_ground_truth():
    # use best self-implemented sampler, which has a chance of correctly
    # sampling from the distribution
    sampler = pypesto.AdaptiveParallelTemperingSampler(
        internal_sampler=pypesto.AdaptiveMetropolisSampler(), n_chains=5)

    problem = gaussian_problem()

    result = pypesto.minimize(problem)

    result = pypesto.sample(problem, n_samples=10000,
                            result=result, sampler=sampler)

    # get samples of first chain
    samples = result.sample_result.trace_x[0].flatten()

    pypesto.visualize.sampling_1d_marginals(result)
    plt.show()

    # test against different distributions

    statistic, pval = kstest(samples, 'norm')
    print(statistic, pval)
    assert statistic < 0.1

    statistic, pval = kstest(samples, 'uniform')
    print(statistic, pval)
    assert statistic > 0.1
