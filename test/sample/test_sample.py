"""Tests for `pypesto.sample` methods."""

import os

import numpy as np
import pytest
import scipy.optimize as so
from scipy.integrate import quad
from scipy.stats import ks_2samp, kstest, multivariate_normal, norm, uniform

import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample
from pypesto.C import OBJECTIVE_NEGLOGLIKE, OBJECTIVE_NEGLOGPOST
from pypesto.objective import (
    AggregatedObjective,
    NegLogParameterPriors,
    Objective,
)


def gaussian_llh(x):
    return float(norm.logpdf(x).item())


def gaussian_nllh_grad(x):
    mu, sigma = 0, 1
    return np.array([((x - mu) / (sigma**2))])


def gaussian_nllh_hess(x):
    sigma = 1
    return np.array([(1 / (sigma**2))])


def gaussian_problem():
    def nllh(x):
        return -gaussian_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(objective=objective, lb=[-10], ub=[10])
    return problem


def gaussian_mixture_llh(x):
    return np.log(
        0.3 * multivariate_normal.pdf(x, mean=-1.5, cov=0.1)
        + 0.7 * multivariate_normal.pdf(x, mean=2.5, cov=0.2)
    )


def gaussian_mixture_problem():
    """Problem based on a mixture of gaussians."""

    def nllh(x):
        return -gaussian_mixture_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(
        objective=objective, lb=[-10], ub=[10], x_names=["x"]
    )
    return problem


def gaussian_mixture_separated_modes_llh(x):
    return np.log(
        0.5 * multivariate_normal.pdf(x, mean=-1.0, cov=0.7)
        + 0.5 * multivariate_normal.pdf(x, mean=100.0, cov=0.8)
    )


def gaussian_mixture_separated_modes_problem():
    """Problem based on a mixture of gaussians with far/separated modes."""

    def nllh(x):
        return -gaussian_mixture_separated_modes_llh(x)

    objective = pypesto.Objective(fun=nllh)
    problem = pypesto.Problem(
        objective=objective, lb=[-100], ub=[200], x_names=["x"]
    )
    return problem


def rosenbrock_problem():
    """Problem based on rosenbrock objective.

    Features
    --------
    * 3-dim
    * has fixed parameters
    * has gradient
    """
    objective = pypesto.Objective(fun=so.rosen, grad=so.rosen_der)

    dim_full = 2
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))

    problem = pypesto.Problem(
        objective=objective,
        lb=lb,
        ub=ub,
        x_fixed_indices=[1],
        x_fixed_vals=[2],
    )
    return problem


def create_petab_problem():
    import petab.v1 as petab

    import pypesto.petab

    current_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(
        os.path.join(current_path, "..", "..", "doc", "example")
    )
    # import to petab
    petab_problem = petab.Problem.from_yaml(
        dir_path + "/conversion_reaction/conversion_reaction.yaml"
    )
    # import to pypesto
    importer = pypesto.petab.PetabImporter(petab_problem)
    # create problem
    problem = importer.create_problem()

    return problem


def sample_petab_problem():
    # create problem
    problem = create_petab_problem()

    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )
    result = sample.sample(
        problem,
        n_samples=1000,
        sampler=sampler,
        x0=np.array([3, -4]),
    )
    return result


def prior(x):
    return multivariate_normal.pdf(x, mean=-1.0, cov=0.7)


def likelihood(x):
    return uniform.pdf(x, loc=-10.0, scale=20.0)[0]


def negative_log_posterior(x):
    return -np.log(likelihood(x)) - np.log(prior(x))


def negative_log_prior(x):
    return -np.log(prior(x))


@pytest.fixture(
    params=[
        "Metropolis",
        "AdaptiveMetropolis",
        "ParallelTempering",
        "AdaptiveParallelTempering",
        "Pymc",
        "Emcee",
        "Dynesty",
    ]
)
def sampler(request):
    if request.param == "Metropolis":
        return sample.MetropolisSampler(
            options={
                "show_progress": False,
            },
        )
    elif request.param == "AdaptiveMetropolis":
        return sample.AdaptiveMetropolisSampler(
            options={
                "show_progress": False,
            },
        )
    elif request.param == "ParallelTempering":
        return sample.ParallelTemperingSampler(
            internal_sampler=sample.MetropolisSampler(),
            options={
                "show_progress": False,
            },
            betas=[1, 1e-2, 1e-4],
        )
    elif request.param == "AdaptiveParallelTempering":
        return sample.AdaptiveParallelTemperingSampler(
            internal_sampler=sample.AdaptiveMetropolisSampler(),
            options={
                "show_progress": False,
            },
            n_chains=5,
        )
    elif request.param == "Pymc":
        from pypesto.sample.pymc import PymcSampler

        return PymcSampler(tune=5, progressbar=False)
    elif request.param == "Emcee":
        return sample.EmceeSampler(nwalkers=10)
    elif request.param == "Dynesty":
        return sample.DynestySampler(objective_type="negloglike")


@pytest.fixture(params=["gaussian", "gaussian_mixture", "rosenbrock"])
def problem(request):
    if request.param == "gaussian":
        return gaussian_problem()
    if request.param == "gaussian_mixture":
        return gaussian_mixture_problem()
    elif request.param == "rosenbrock":
        return rosenbrock_problem()


def test_pipeline(sampler, problem):
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
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=100,
        result=result,
    )
    # remove warnings in test/sample/test_sample.
    # Warning here: pypesto/visualize/sampling.py:1104
    # geweke test
    sample.geweke_test(result=result)


def test_ground_truth():
    """Test whether we actually retrieve correct distributions."""
    # use best self-implemented sampler, which has a chance of correctly
    # sample from the distribution
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        options={
            "show_progress": False,
        },
        n_chains=5,
    )

    problem = gaussian_problem()

    result = optimize.minimize(
        problem,
        progress_bar=False,
    )

    result = sample.sample(
        problem,
        n_samples=5000,
        result=result,
        sampler=sampler,
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


def test_ground_truth_separated_modes():
    """Test whether we actually retrieve correct distributions."""
    # use best self-implemented sampler, which has a chance to correctly
    # sample from the distribution

    # First use parallel tempering with 3 chains
    sampler = sample.AdaptiveParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        options={
            "show_progress": False,
        },
        n_chains=3,
    )

    problem = gaussian_mixture_separated_modes_problem()

    result = sample.sample(
        problem,
        n_samples=1e4,
        sampler=sampler,
        x0=np.array([0.0]),
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # generate bimodal ground-truth samples
    # "first" mode centered at -1
    rvs1 = norm.rvs(size=5000, loc=-1.0, scale=np.sqrt(0.7))
    # "second" mode centered at 100
    rvs2 = norm.rvs(size=5001, loc=100.0, scale=np.sqrt(0.8))

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]), samples)

    # only parallel tempering finds both modes
    print(statistic, pval)
    assert statistic < 0.2

    # sample using adaptive metropolis (single-chain)
    # initiated around the "first" mode of the distribution
    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )
    result = sample.sample(
        problem,
        n_samples=1e4,
        sampler=sampler,
        x0=np.array([-2.0]),
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]), samples)

    # single-chain adaptive metropolis does not find both modes
    print(statistic, pval)
    assert statistic > 0.1

    # actually centered at the "first" mode
    statistic, pval = ks_2samp(rvs1, samples)

    print(statistic, pval)
    assert statistic < 0.1

    # sample using adaptive metropolis (single-chain)
    # initiated around the "second" mode of the distribution
    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )
    result = sample.sample(
        problem,
        n_samples=1e4,
        sampler=sampler,
        x0=np.array([120.0]),
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # test for distribution similarity
    statistic, pval = ks_2samp(np.concatenate([rvs1, rvs2]), samples)

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
    sampler = sample.ParallelTemperingSampler(
        internal_sampler=sample.MetropolisSampler(),
        options={
            "show_progress": False,
        },
        n_chains=2,
    )
    result = sample.sample(
        problem,
        n_samples=10,
        x0=x0s,
        sampler=sampler,
    )

    assert result.sample_result.trace_neglogpost.shape[0] == 2
    assert [
        result.sample_result.trace_x[0][0],
        result.sample_result.trace_x[1][0],
    ] == x0s


def test_regularize_covariance():
    """
    Make sure that `regularize_covariance` renders symmetric matrices
    positive definite.
    """
    matrix = np.array([[-1.0, -4.0], [-4.0, 1.0]])
    assert np.any(np.linalg.eigvals(matrix) < 0)
    reg = sample.adaptive_metropolis.regularize_covariance(matrix, 1e-6)
    assert np.all(np.linalg.eigvals(reg) >= 0)


def test_geweke_test_switch():
    """Check geweke test returns expected burn in index."""
    warm_up = np.zeros((100, 2))
    converged = np.ones((901, 2))
    chain = np.concatenate((warm_up, converged), axis=0)
    burn_in = sample.diagnostics.burn_in_by_sequential_geweke(chain=chain)
    assert burn_in == 100


def test_geweke_test_switch_short():
    """Check geweke test returns expected burn in index
    for small sample numbers."""
    warm_up = np.zeros((25, 2))
    converged = np.ones((75, 2))
    chain = np.concatenate((warm_up, converged), axis=0)
    burn_in = sample.diagnostics.burn_in_by_sequential_geweke(chain=chain)
    assert burn_in == 25


def test_geweke_test_unconverged():
    """Check that the geweke test reacts nicely to small sample numbers."""
    problem = gaussian_problem()

    sampler = sample.MetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    # optimization
    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem,
        sampler=sampler,
        n_samples=100,
        result=result,
    )

    # run geweke test (should not fail!)
    sample.geweke_test(result)


def test_autocorrelation_pipeline():
    """Check that the autocorrelation test works."""
    problem = gaussian_problem()

    sampler = sample.MetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    # optimization
    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=1000,
        result=result,
    )

    # run auto-correlation with previous geweke
    sample.geweke_test(result)

    ac1 = sample.auto_correlation(result)

    # run auto-correlation without previous geweke
    result.sample_result.burn_in = None
    ac2 = sample.auto_correlation(result)

    assert ac1 == ac2

    # run effective sample size with previous geweke
    # and autocorrelation
    ess1 = sample.effective_sample_size(result)

    # run effective sample size without previous geweke
    # and autocorrelation
    result.sample_result.burn_in = None
    result.sample_result.auto_correlation = None
    ess2 = sample.effective_sample_size(result)

    assert ess1 == ess2


def test_autocorrelation_short_chain():
    """Check that the autocorrelation
    reacts nicely to small sample numbers."""
    problem = gaussian_problem()

    sampler = sample.MetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    # optimization
    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem,
        sampler=sampler,
        n_samples=10,
        result=result,
    )

    # manually set burn in to chain length (only for testing!!)
    chain_length = result.sample_result.trace_x.shape[1]
    result.sample_result.burn_in = chain_length

    # run auto-correlation
    ac = sample.auto_correlation(result)

    assert ac is None

    # run effective sample size
    ess = sample.effective_sample_size(result)

    assert ess is None


@pytest.mark.flaky(reruns=3)
def test_autocorrelation_mixture():
    """Check that the autocorrelation is the same for the same chain
    with different scalings."""
    chain = np.array(np.random.randn(101, 2))

    auto_correlation_1 = sample.diagnostics.autocorrelation_sokal(chain=chain)
    auto_correlation_2 = sample.diagnostics.autocorrelation_sokal(
        chain=2 * chain
    )
    auto_correlation_3 = sample.diagnostics.autocorrelation_sokal(
        chain=-3 * chain
    )

    assert (abs(auto_correlation_1 - auto_correlation_2) < 1e-15).all()
    assert (abs(auto_correlation_2 - auto_correlation_3) < 1e-15).all()
    assert (abs(auto_correlation_1 - auto_correlation_3) < 1e-15).all()


def test_autocorrelation_dim():
    """Check that the autocorrelation returns as
    many entries as parameters."""
    # Loop over different sizes of parameter vectors
    for n in range(4):
        # create the chain for n parameters
        chain = np.array(np.random.randn(101, n + 1))
        # calculate the autocorrelation
        auto_correlation = sample.diagnostics.autocorrelation_sokal(
            chain=chain
        )
        assert len(auto_correlation) == (n + 1)


def test_autocorrelation_high():
    """Check that the autocorrelation is high for a not well-mixed chain."""
    # there should be always need to be some variability
    chain = np.concatenate(
        (np.ones((50, 1)), 2 * np.ones((35, 1)), np.ones((25, 1)))
    )

    auto_correlation = sample.diagnostics.autocorrelation_sokal(chain=chain)

    assert auto_correlation > 10


def test_empty_prior():
    """Check that priors are zero when none are defined."""
    # define negative log posterior
    posterior_fun = pypesto.Objective(fun=negative_log_posterior)

    # define pypesto problem without prior object
    test_problem = pypesto.Problem(
        objective=posterior_fun, lb=-10, ub=10, x_names=["x"]
    )

    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    result = sample.sample(
        test_problem,
        n_samples=50,
        sampler=sampler,
        x0=np.array([0.0]),
    )

    # get log prior values of first chain
    logprior_trace = -result.sample_result.trace_neglogprior[0, :]

    # check that all entries are zero
    assert (logprior_trace == 0.0).all()


@pytest.mark.flaky(reruns=2)
def test_prior():
    """Check that priors are defined for sampling."""
    # define negative log posterior
    posterior_fun = pypesto.Objective(fun=negative_log_posterior)

    # define negative log prior
    prior_fun = pypesto.Objective(fun=negative_log_prior)

    # define pypesto prior object
    prior_object = pypesto.NegLogPriors(objectives=[prior_fun])

    # define pypesto problem using prior object
    test_problem = pypesto.Problem(
        objective=posterior_fun,
        x_priors_defs=prior_object,
        lb=-10,
        ub=10,
        x_names=["x"],
    )

    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    result = sample.sample(
        test_problem,
        n_samples=1e4,
        sampler=sampler,
        x0=np.array([0.0]),
    )

    # get log prior values of first chain
    logprior_trace = -result.sample_result.trace_neglogprior[0, :]

    # check that not all entries are zero
    assert (logprior_trace != 0.0).any()

    # get samples of first chain
    samples = result.sample_result.trace_x[0, :, 0]

    # generate ground-truth samples
    rvs = norm.rvs(size=5000, loc=-1.0, scale=np.sqrt(0.7))

    # check sample distribution agreement with the ground-truth
    statistic, pval = ks_2samp(rvs, samples)
    print(statistic, pval)

    assert statistic < 0.1


def test_samples_cis():
    """
    Test whether :py:func:`pypesto.sample.calculate_ci_mcmc_sample` produces
    percentile-based credibility intervals correctly.
    """
    # load problem
    problem = gaussian_problem()

    # set a sampler
    sampler = sample.MetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    # optimization
    result = optimize.minimize(
        problem=problem,
        n_starts=3,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=1000,
        result=result,
    )

    # manually set burn in (only for testing!!)
    burn_in = 100
    result.sample_result.burn_in = burn_in

    # get converged chain
    converged_chain = np.asarray(
        result.sample_result.trace_x[0, result.sample_result.burn_in :, :]
    )

    # set confidence levels
    alpha_values = [0.99, 0.95, 0.68]

    # loop over confidence levels
    for alpha in alpha_values:
        # calculate parameter samples confidence intervals
        lb, ub = sample.calculate_ci_mcmc_sample(result, ci_level=alpha)
        # get corresponding percentiles to alpha
        percentiles = 100 * np.array([(1 - alpha) / 2, 1 - (1 - alpha) / 2])
        # check result agreement
        diff = np.percentile(converged_chain, percentiles, axis=0) - [lb, ub]

        assert (diff == 0).all()
        # check if lower bound is smaller than upper bound
        assert (lb < ub).all()
        # check if dimensions agree
        assert lb.shape == ub.shape


def test_dynesty_mcmc_samples():
    problem = gaussian_problem()
    sampler = sample.DynestySampler(objective_type=OBJECTIVE_NEGLOGLIKE)

    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=None,
        filename=None,
    )

    original_sample_result = sampler.get_original_samples()
    mcmc_sample_result = result.sample_result

    # Nested sampling function values are monotonically increasing
    assert (np.diff(original_sample_result.trace_neglogpost) <= 0).all()
    # MCMC samples are not
    assert not (np.diff(mcmc_sample_result.trace_neglogpost) <= 0).all()


def test_dynesty_posterior():
    # define negative log posterior
    posterior_fun = pypesto.Objective(fun=negative_log_posterior)

    # define negative log prior
    prior_fun = pypesto.Objective(fun=negative_log_prior)

    # define pypesto prior object
    prior_object = pypesto.NegLogPriors(objectives=[prior_fun])

    # define pypesto problem using prior object
    test_problem = pypesto.Problem(
        objective=posterior_fun,
        x_priors_defs=prior_object,
        lb=-10,
        ub=10,
        x_names=["x"],
    )

    # define sampler
    sampler = sample.DynestySampler(
        objective_type=OBJECTIVE_NEGLOGPOST
    )  # default

    result = sample.sample(
        problem=test_problem,
        sampler=sampler,
        n_samples=None,
        filename=None,
    )

    original_sample_result = sampler.get_original_samples()
    mcmc_sample_result = result.sample_result

    # Nested sampling function values are monotonically increasing
    assert (np.diff(original_sample_result.trace_neglogpost) <= 0).all()
    # MCMC samples are not
    assert not (np.diff(mcmc_sample_result.trace_neglogpost) <= 0).all()


@pytest.mark.flaky(reruns=2)  # sometimes not all chains converge
def test_thermodynamic_integration():
    # test thermodynamic integration
    problem = gaussian_problem()

    # approximation should be better for more chains
    n_chains = 10
    tol = 1
    sampler = sample.ParallelTemperingSampler(
        internal_sampler=sample.AdaptiveMetropolisSampler(),
        options={"show_progress": False, "beta_init": "beta_decay"},
        n_chains=n_chains,
    )

    result = optimize.minimize(
        problem,
        progress_bar=False,
    )

    result = sample.sample(
        problem,
        n_samples=2000,
        result=result,
        sampler=sampler,
    )

    # compute the log evidence using trapezoid and simpson rule
    log_evidence = sampler.compute_log_evidence(result, method="trapezoid")
    log_evidence_not_all = sampler.compute_log_evidence(
        result, method="trapezoid", use_all_chains=False
    )
    log_evidence_simps = sampler.compute_log_evidence(result, method="simpson")

    # use steppingstone sampling
    log_evidence_steppingstone = sampler.compute_log_evidence(
        result, method="steppingstone"
    )

    # compute evidence
    evidence = quad(
        lambda x: 1
        / (problem.ub[0] - problem.lb[0])
        * np.exp(gaussian_llh(x)),
        a=problem.lb[0],
        b=problem.ub[0],
    )

    # compare to known value
    assert np.isclose(log_evidence, np.log(evidence[0]), atol=tol)
    assert np.isclose(log_evidence_not_all, np.log(evidence[0]), atol=tol)
    assert np.isclose(log_evidence_simps, np.log(evidence[0]), atol=tol)
    assert np.isclose(
        log_evidence_steppingstone, np.log(evidence[0]), atol=tol
    )


def test_laplace_approximation_log_evidence():
    """Test the laplace approximation of the log evidence."""
    log_evidence_true = -1.15  # approximated by hand

    problem = create_petab_problem()

    # hess
    result = optimize.minimize(
        problem=problem,
        n_starts=10,
        progress_bar=False,
    )
    log_evidence = sample.estimate_evidence.laplace_approximation_log_evidence(
        problem, result.optimize_result.x[0]
    )
    assert np.isclose(log_evidence, log_evidence_true, atol=0.1)


@pytest.mark.flaky(reruns=2)
def test_harmonic_mean_log_evidence():
    tol = 1
    # define problem
    problem = gaussian_problem()

    # run optimization and MCMC
    result = optimize.minimize(
        problem,
        progress_bar=False,
        n_starts=10,
    )
    result = sample.sample(
        problem,
        n_samples=2000,
        result=result,
    )

    # compute the log evidence using harmonic mean
    harmonic_evidence = sample.estimate_evidence.harmonic_mean_log_evidence(
        result
    )
    # compute the log evidence using stabilized harmonic mean
    prior_samples = np.random.uniform(problem.lb, problem.ub, size=100)
    harmonic_stabilized_evidence = (
        sample.estimate_evidence.harmonic_mean_log_evidence(
            result=result,
            prior_samples=prior_samples,
            neg_log_likelihood_fun=problem.objective,
        )
    )

    # compute real evidence
    evidence = quad(
        lambda x: 1
        / (problem.ub[0] - problem.lb[0])
        * np.exp(gaussian_llh(x)),
        a=problem.lb[0],
        b=problem.ub[0],
    )

    # compare to known value
    assert np.isclose(harmonic_evidence, np.log(evidence[0]), atol=tol)
    assert np.isclose(
        harmonic_stabilized_evidence, np.log(evidence[0]), atol=tol
    )


@pytest.mark.flaky(reruns=2)
def test_bridge_sampling():
    # define problem
    objective = Objective(
        fun=lambda x: -gaussian_llh(x),
        grad=gaussian_nllh_grad,
        hess=gaussian_nllh_hess,
    )
    prior_true = NegLogParameterPriors(
        [
            {
                "index": 0,
                "density_fun": lambda x: (1 / (10 + 10)),
                "density_dx": lambda x: 0,
                "density_ddx": lambda x: 0,
            },
        ]
    )
    problem = pypesto.Problem(
        objective=AggregatedObjective([objective, prior_true]),
        lb=[-10],
        ub=[10],
        x_names=["x"],
    )

    # run optimization and MCMC
    result = optimize.minimize(problem, progress_bar=False, n_starts=10)
    result = sample.sample(
        problem,
        n_samples=1000,
        result=result,
    )

    # compute the log evidence using harmonic mean
    bridge_log_evidence = sample.estimate_evidence.bridge_sampling(result)
    assert isinstance(bridge_log_evidence, float)
