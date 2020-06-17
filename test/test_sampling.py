"""
This is for testing optimization of the pypesto.Objective.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm, kstest, ks_2samp, uniform
import scipy.optimize as so
import matplotlib.pyplot as plt
import pytest

import pypesto
from pypesto.objective import NegLogPriors


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


def gaussian_mixture_separated_modes_llh(x):
    return np.log(
        0.5*multivariate_normal.pdf(x, mean=-1., cov=0.7)
        + 0.5*multivariate_normal.pdf(x, mean=100., cov=0.8))


def gaussian_mixture_separated_modes_problem():
    """Problem based on a mixture of gaussians with far/separated modes."""
    def nllh(x):
        return - gaussian_mixture_separated_modes_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(objective=objective, lb=[-100], ub=[200],
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


def prior(x):
    return multivariate_normal.pdf(x, mean=-1., cov=0.7)


def likelihood(x):
    return uniform.pdf(x, loc=-10., scale=20.)[0]


def negative_log_posterior(x):
    return - np.log(likelihood(x)) - np.log(prior(x))


def negative_log_prior(x):
    return - np.log(prior(x))


@pytest.fixture(params=['Metropolis',
                        'AdaptiveMetropolis',
                        'ParallelTempering',
                        'AdaptiveParallelTempering',
                        'Pymc3'])
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
    elif request.param == 'Pymc3':
        return pypesto.Pymc3Sampler(tune=5)


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
        problem, sampler=sampler, n_samples=100, result=result)

    # some plot
    pypesto.visualize.sampling_1d_marginals(result)
    plt.close()


def test_ground_truth():
    """Test whether we actually retrieve correct distributions."""
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

    # test against different distributions

    statistic, pval = kstest(samples, 'norm')
    print(statistic, pval)
    assert statistic < 0.1

    statistic, pval = kstest(samples, 'uniform')
    print(statistic, pval)
    assert statistic > 0.1


def test_ground_truth_separated_modes():
    """Test whether we actually retrieve correct distributions."""
    # use best self-implemented sampler, which has a chance of correctly
    # sampling from the distribution

    # First use parallel tempering with 3 chains
    sampler = pypesto.AdaptiveParallelTemperingSampler(
        internal_sampler=pypesto.AdaptiveMetropolisSampler(), n_chains=3)

    problem = gaussian_mixture_separated_modes_problem()

    result = pypesto.sample(problem, n_samples=1e4,
                            sampler=sampler,
                            x0=np.array([0.]))

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # generate bimodal ground-truth samples
    # "first" mode centered at -1
    rvs1 = norm.rvs(size=5000, loc=-1., scale=np.sqrt(0.7))
    # "second" mode centered at 100
    rvs2 = norm.rvs(size=5001, loc=100., scale=np.sqrt(0.8))

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]),
                               samples)

    # only parallel tempering finds both modes
    print(statistic, pval)
    assert statistic < 0.1

    # sample using adaptive metropolis (single-chain)
    # initiated around the "first" mode of the distribution
    sampler = pypesto.AdaptiveMetropolisSampler()
    result = pypesto.sample(problem, n_samples=1e4,
                            sampler=sampler,
                            x0=np.array([-2.]))

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]),
                               samples)

    # single-chain adaptive metropolis does not find both modes
    print(statistic, pval)
    assert statistic > 0.1

    # actually centered at the "first" mode
    statistic, pval = ks_2samp(rvs1, samples)

    print(statistic, pval)
    assert statistic < 0.1

    # sample using adaptive metropolis (single-chain)
    # initiated around the "second" mode of the distribution
    sampler = pypesto.AdaptiveMetropolisSampler()
    result = pypesto.sample(problem, n_samples=1e4,
                            sampler=sampler,
                            x0=np.array([120.]))

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]),
                               samples)

    # single-chain adaptive metropolis does not find both modes
    print(statistic, pval)
    assert statistic > 0.1

    # actually centered at the "second" mode
    statistic, pval = ks_2samp(rvs2, samples)

    print(statistic, pval)
    assert statistic < 0.1


def test_multiple_startpoints():
    problem = gaussian_problem()
    x0s = [np.array([0]), np.array([1])]
    sampler = pypesto.ParallelTemperingSampler(
        internal_sampler=pypesto.MetropolisSampler(),
        n_chains=2
    )
    result = pypesto.sample(problem, n_samples=10, x0=x0s, sampler=sampler)

    assert result.sample_result.trace_neglogpost.shape[0] == 2
    assert [result.sample_result.trace_x[0][0],
            result.sample_result.trace_x[1][0]] == x0s


def test_regularize_covariance():
    """
    Make sure that `regularize_covariance` renders symmetric matrices
    positive definite.
    """
    matrix = np.array([[-1., -4.], [-4., 1.]])
    assert np.any(np.linalg.eigvals(matrix) < 0)
    reg = pypesto.sampling.adaptive_metropolis.regularize_covariance(
        matrix, 1e-6)
    assert np.all(np.linalg.eigvals(reg) >= 0)


def test_geweke_test_switch():
    """Check geweke test returns expected burn in index."""
    warm_up = np.zeros((100, 2))
    converged = np.ones((901, 2))
    chain = np.concatenate((warm_up, converged), axis=0)
    burn_in = pypesto.sampling.diagnostics.burn_in_by_sequential_geweke(
        chain=chain)
    assert burn_in == 100


def test_geweke_test_switch_short():
    """Check geweke test returns expected burn in index
    for small sample numbers."""
    warm_up = np.zeros((25, 2))
    converged = np.ones((75, 2))
    chain = np.concatenate((warm_up, converged), axis=0)
    burn_in = pypesto.sampling.diagnostics.burn_in_by_sequential_geweke(
        chain=chain)
    assert burn_in == 25


def test_geweke_test_unconverged():
    """Check that the geweke test reacts nicely to small sample numbers."""
    problem = gaussian_problem()

    sampler = pypesto.MetropolisSampler()

    # optimization
    result = pypesto.minimize(problem, n_starts=3)

    # sampling
    result = pypesto.sample(
        problem, sampler=sampler, n_samples=100, result=result)

    # run geweke test (should not fail!)
    pypesto.sampling.geweke_test(result)


def test_empty_prior():
    """Check that priors are zero when none are defined."""
    # define negative log posterior
    posterior_fun = pypesto.Objective(fun=negative_log_posterior)

    # define pypesto problem without prior object
    test_problem = pypesto.Problem(objective=posterior_fun,
                                   lb=-10, ub=10,
                                   x_names=['x'])

    sampler = pypesto.AdaptiveMetropolisSampler()

    result = pypesto.sample(test_problem, n_samples=50,
                            sampler=sampler,
                            x0=np.array([0.]))

    # get log prior values of first chain
    logprior_trace = -result.sample_result.trace_neglogprior[0, :]

    # check that all entries are zero
    assert (logprior_trace == 0.).all()


def test_prior():
    """Check that priors are defined in sampling."""
    # define negative log posterior
    posterior_fun = pypesto.Objective(fun=negative_log_posterior)

    # define negative log prior
    prior_fun = pypesto.Objective(fun=negative_log_prior)

    # define pypesto prior object
    prior_object = NegLogPriors(objectives=[prior_fun])

    # define pypesto problem using prior object
    test_problem = pypesto.Problem(objective=posterior_fun,
                                   x_priors_defs=prior_object,
                                   lb=-10, ub=10,
                                   x_names=['x'])

    sampler = pypesto.AdaptiveMetropolisSampler()

    result = pypesto.sample(test_problem, n_samples=1e4,
                            sampler=sampler,
                            x0=np.array([0.]))

    # get log prior values of first chain
    logprior_trace = -result.sample_result.trace_neglogprior[0, :]

    # check that not all entries are zero
    assert (logprior_trace != 0.).any()

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # generate ground-truth samples
    rvs = norm.rvs(size=5000, loc=-1., scale=np.sqrt(0.7))

    # check sample distribution agreement with the ground-truth
    statistic, pval = ks_2samp(rvs, samples)
    print(statistic, pval)

    assert statistic < 0.1
