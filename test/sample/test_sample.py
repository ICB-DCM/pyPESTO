"""Tests for `pypesto.sample` methods."""

import importlib
import logging
import warnings

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import ks_2samp, kstest, norm

import pypesto
import pypesto.optimize as optimize
import pypesto.sample as sample
from pypesto.C import OBJECTIVE_NEGLOGLIKE, OBJECTIVE_NEGLOGPOST
from pypesto.objective import (
    AggregatedObjective,
    NegLogParameterPriors,
    Objective,
)

from .util import (
    LB_GAUSSIAN,
    N_CHAINS,
    N_SAMPLE_FEW,
    N_SAMPLE_MANY,
    N_SAMPLE_SOME,
    N_STARTS_FEW,
    N_STARTS_SOME,
    STATISTIC_TOL,
    UB_GAUSSIAN,
    X_NAMES,
    create_petab_problem,
    gaussian_llh,
    gaussian_mixture_problem,
    gaussian_nllh_grad,
    gaussian_nllh_hess,
    gaussian_problem,
    negative_log_posterior,
    negative_log_prior,
    rosenbrock_problem,
)


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
            n_chains=N_CHAINS,
        )
    elif request.param == "Pymc":
        from pypesto.sample.pymc import PymcSampler

        return PymcSampler(tune=5, progressbar=False, chains=N_CHAINS)
    elif request.param == "Emcee":
        return sample.EmceeSampler(nwalkers=10)
    elif request.param == "Dynesty":
        return sample.DynestySampler(
            objective_type=OBJECTIVE_NEGLOGLIKE,
            run_args={"maxiter": N_SAMPLE_FEW},
        )


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
        n_starts=N_STARTS_FEW,
        optimizer=optimizer,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=N_SAMPLE_FEW,
        result=result,
    )
    # test dynesty mcmc samples
    if isinstance(sampler, sample.DynestySampler):
        trace_original = sampler.get_original_samples().trace_neglogpost
        trace_mcmc = result.sample_result.trace_neglogpost
        # Nested sampling function values are monotonically increasing
        assert (np.diff(trace_original) <= 0).all()
        # MCMC samples are not
        assert not (np.diff(trace_mcmc) <= 0).all()
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
        n_chains=N_CHAINS,
    )

    problem = gaussian_problem()

    result = optimize.minimize(
        problem,
        progress_bar=False,
        n_starts=N_STARTS_SOME,
    )

    result = sample.sample(
        problem,
        n_samples=N_SAMPLE_MANY,
        result=result,
        sampler=sampler,
    )

    # get samples of first chain
    samples = result.sample_result.trace_x[0].flatten()

    # test against different distributions

    statistic, pval = kstest(samples, "norm")
    assert statistic < STATISTIC_TOL

    statistic, pval = kstest(samples, "uniform")
    assert statistic > STATISTIC_TOL


def test_multiple_startpoints():
    problem = gaussian_problem()
    x0s = [np.array([0]), np.array([1]), np.array([2])]
    sampler = sample.ParallelTemperingSampler(
        internal_sampler=sample.MetropolisSampler(),
        options={
            "show_progress": False,
        },
        n_chains=N_CHAINS,
    )
    result = sample.sample(
        problem,
        n_samples=N_SAMPLE_FEW,
        x0=x0s,
        sampler=sampler,
    )

    assert result.sample_result.trace_neglogpost.shape[0] == N_CHAINS
    assert [
        result.sample_result.trace_x[0][0],
        result.sample_result.trace_x[1][0],
        result.sample_result.trace_x[2][0],
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


geweke = importlib.import_module("pypesto.sample.geweke_test")


def _ar1(n: int, rho: float, rng: np.random.Generator, mean: float = 5.0):
    """Stationary AR(1) chain with unit marginal variance, shape (n, 1)."""
    noise = rng.normal(size=n) * np.sqrt(1 - rho**2)
    x = np.empty(n)
    x[0] = rng.normal()
    for t in range(1, n):
        x[t] = rho * x[t - 1] + noise[t]
    return (x + mean)[:, np.newaxis]


def _reference_welch_at_zero(x: np.ndarray, nw: int) -> float:
    """Frozen copy of the previous FFT-based `spectrum(x, len(x), nw)[0]`."""
    n_overlap = nw // 2
    w = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, nw + 1) / (nw + 1)))
    k = (len(x) - n_overlap) // (nw - n_overlap)
    index = np.arange(nw)
    density = np.zeros(len(x))
    for _ in range(k):
        density += np.abs(np.fft.fft(w * x[index], n=len(x))) ** 2
        index += nw - n_overlap
    return density[0] / (k * np.linalg.norm(w) ** 2)


@pytest.mark.parametrize("n", [8, 9, 50, 99, 100, 1000])
def test_geweke_spectral_density_matches_welch(n):
    """The direct frequency-zero estimate equals Welch's method at bin 0."""
    x = np.random.default_rng(n).normal(3, 1, n)
    nw = n // 4
    assert np.isclose(
        geweke._spectral_density_at_zero(x[:, np.newaxis], nw)[0],
        _reference_welch_at_zero(x, nw),
    )


def test_geweke_spectrum0_estimates_long_run_variance():
    """Centered, sqrt(n)-batch estimate of S(0) converges (S(0) = 1)."""
    rng = np.random.default_rng(0)
    estimates = [
        geweke.spectrum0(rng.normal(5, 1, (10_000, 1))).item()
        for _ in range(200)
    ]
    assert abs(np.mean(estimates) - 1) < 0.05
    assert np.std(estimates) < 0.15


def test_geweke_zscore_invariant_to_shift_and_scale():
    """z must not depend on where the parameter sits or on its units."""
    chain = _ar1(1000, 0.5, np.random.default_rng(1))
    z, _ = geweke.calculate_zscore(chain)
    assert np.allclose(geweke.calculate_zscore(chain + 1000)[0], z)
    assert np.allclose(geweke.calculate_zscore(chain * 37)[0], z)


def test_geweke_zscore_segments(monkeypatch):
    """The first floor(a*n) and the last floor(b*n) samples are compared."""
    seen = []

    def record(segment):
        seen.append(segment[:, 0].copy())
        return np.ones((1, segment.shape[1]))

    monkeypatch.setattr(geweke, "spectrum0", record)
    chain = np.arange(1000, dtype=float)[:, np.newaxis]
    geweke.calculate_zscore(chain, a=0.1, b=0.5)
    assert np.array_equal(seen[0], np.arange(100))
    assert np.array_equal(seen[1], np.arange(500, 1000))


@pytest.mark.parametrize(
    "a, b, valid",
    [(0.5, 0.5, True), (0.4, 0.6, True), (0.6, 0.5, False), (0, 0.5, False)]
    + [(0.1, 1.2, False)],
)
def test_geweke_zscore_fraction_validation(a, b, valid):
    """Fractions must lie in (0, 1) and must not overlap."""
    chain = np.random.default_rng(2).normal(size=(100, 1))
    if valid:
        geweke.calculate_zscore(chain, a=a, b=b)
    else:
        with pytest.raises(ValueError):
            geweke.calculate_zscore(chain, a=a, b=b)


def test_geweke_zscore_degenerate_segments():
    """Constant segments give z = 0 (equal) or +-inf (different)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        equal, _ = geweke.calculate_zscore(np.ones((100, 1)))
        different, _ = geweke.calculate_zscore(
            np.r_[np.zeros(50), np.ones(50)][:, np.newaxis]
        )
    assert equal.item() == 0
    assert different.item() == -np.inf


def test_geweke_spectrum0_not_inflated_by_drift():
    """Centering each Geweke window, not the whole chain, keeps S(0) small."""
    rng = np.random.default_rng(3)
    chain = 5 + 0.003 * np.arange(1000)[:, np.newaxis]
    chain += rng.normal(size=(1000, 1))
    assert geweke.spectrum0(chain[:100]).item() < 2


@pytest.mark.parametrize(
    "max_abs_z, expected",
    [
        ([0.1] * 20, 0),  # nothing significant
        ([10, 10] + [0.1] * 18, 2),  # first two starting points rejected
        ([10] * 20, None),  # everything rejected
        ([3.06] + [0.0] * 19, 1),  # just above Holm's 1st threshold (3.05)
        ([3.04] + [0.0] * 19, 0),  # just below it
        ([5, 2.1, 2.2], 1),  # step-down stops at 2.2, so 2.1 is retained
        ([3, 1, 2], 1),  # case the previous rank mix-up got wrong
        ([np.nan, 0.1], 1),  # NaN (too short to test) counts as rejected
    ],
)
def test_geweke_holm_step_down(max_abs_z, expected):
    """Holm's procedure on the per-fragment p-values."""
    assert (
        geweke._first_not_rejected(np.array(max_abs_z, dtype=float), 2.0)
        == expected
    )


def test_geweke_burn_in_iid():
    """A stationary chain has no burn-in."""
    chain = np.random.default_rng(4).normal(5, 1, (1000, 2))
    assert sample.diagnostics.burn_in_by_sequential_geweke(chain) == 0


def test_geweke_burn_in_transient():
    """A decaying initial transient (gone by ~250) is detected."""
    rng = np.random.default_rng(5)
    chain = 5 + rng.normal(size=(1000, 1))
    chain[:300] += 6 * np.exp(-np.arange(300) / 50)[:, np.newaxis]
    burn_in = sample.diagnostics.burn_in_by_sequential_geweke(chain)
    assert 50 <= burn_in <= 250


def test_geweke_burn_in_drift():
    """A drifting chain gets a large burn-in."""
    rng = np.random.default_rng(6)
    chain = 5 + 0.003 * np.arange(1000)[:, np.newaxis]
    chain += rng.normal(size=(1000, 1))
    assert sample.diagnostics.burn_in_by_sequential_geweke(chain) >= 500


def test_geweke_burn_in_false_alarm_rate():
    """Stationary chains rarely get a burn-in (family-wise level ~0.05)."""
    rng = np.random.default_rng(7)
    burn_ins = [
        sample.diagnostics.burn_in_by_sequential_geweke(_ar1(500, 0.3, rng))
        for _ in range(100)
    ]
    assert np.mean(np.array(burn_ins) > 0) <= 0.12


@pytest.mark.parametrize("n", [1, 10, 19])
def test_geweke_burn_in_short_chain(n, caplog):
    """Chains shorter than the number of fragments are 'not converged'."""
    chain = np.random.default_rng(n).normal(size=(n, 2))
    with caplog.at_level(logging.WARNING):
        assert sample.diagnostics.burn_in_by_sequential_geweke(chain) == n
    assert "not have converged" in caplog.text


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
        n_starts=N_STARTS_FEW,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem,
        sampler=sampler,
        n_samples=N_SAMPLE_FEW,
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
        n_starts=N_STARTS_FEW,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=N_SAMPLE_SOME,
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
        n_starts=N_STARTS_FEW,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem,
        sampler=sampler,
        n_samples=N_SAMPLE_FEW,
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
        n_samples=N_SAMPLE_FEW,
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
        lb=LB_GAUSSIAN,
        ub=UB_GAUSSIAN,
        x_names=X_NAMES,
    )

    sampler = sample.AdaptiveMetropolisSampler(
        options={
            "show_progress": False,
        },
    )

    result = sample.sample(
        test_problem,
        n_samples=N_SAMPLE_MANY,
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
    assert statistic < STATISTIC_TOL


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
        n_starts=N_STARTS_FEW,
        progress_bar=False,
    )

    # sample
    result = sample.sample(
        problem=problem,
        sampler=sampler,
        n_samples=N_SAMPLE_SOME,
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
        lb=LB_GAUSSIAN,
        ub=UB_GAUSSIAN,
        x_names=X_NAMES,
    )

    # define sampler
    sampler = sample.DynestySampler(
        objective_type=OBJECTIVE_NEGLOGPOST,
        run_args={"maxiter": N_SAMPLE_FEW},
    )  # default

    result = sample.sample(
        problem=test_problem,
        sampler=sampler,
        n_samples=None,
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
    tol = 2
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
        n_samples=2 * N_SAMPLE_SOME,
        result=result,
        sampler=sampler,
    )

    # compute the log evidence using trapezoid and simpson rule
    log_evidence = sample.evidence.parallel_tempering_log_evidence(
        result, method="trapezoid"
    )
    log_evidence_not_all = sample.evidence.parallel_tempering_log_evidence(
        result, method="trapezoid", use_all_chains=False
    )
    log_evidence_simps = sample.evidence.parallel_tempering_log_evidence(
        result, method="simpson"
    )

    # use steppingstone sampling
    log_evidence_steppingstone = (
        sample.evidence.parallel_tempering_log_evidence(
            result, method="steppingstone"
        )
    )

    # harmonic mean log evidence
    harmonic_evidence = sample.evidence.harmonic_mean_log_evidence(result)
    # compute the log evidence using stabilized harmonic mean
    prior_samples = np.random.uniform(problem.lb, problem.ub, size=100)
    harmonic_stabilized_evidence = sample.evidence.harmonic_mean_log_evidence(
        result=result,
        prior_samples=prior_samples,
        neg_log_likelihood_fun=problem.objective,
    )

    # compute evidence
    evidence = quad(
        lambda x: (
            1 / (problem.ub[0] - problem.lb[0]) * np.exp(gaussian_llh(x))
        ),
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
    assert np.isclose(harmonic_evidence, np.log(evidence[0]), atol=tol)
    assert np.isclose(
        harmonic_stabilized_evidence, np.log(evidence[0]), atol=tol
    )


def test_laplace_approximation_log_evidence():
    """Test the laplace approximation of the log evidence."""
    log_evidence_true = 21.2  # approximated by hand

    problem = create_petab_problem()

    # hess
    result = optimize.minimize(
        problem=problem,
        n_starts=N_STARTS_SOME,
        progress_bar=False,
    )
    log_evidence = sample.evidence.laplace_approximation_log_evidence(
        problem, result.optimize_result.x[0]
    )
    assert np.isclose(log_evidence, log_evidence_true, atol=0.1)


@pytest.mark.flaky(reruns=3)
def test_bridge_sampling():
    tol = 2
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
                "density_fun": lambda x: 1 / (10 + 10),
                "density_dx": lambda x: 0,
                "density_ddx": lambda x: 0,
            },
        ]
    )
    problem = pypesto.Problem(
        objective=AggregatedObjective([objective, prior_true]),
        lb=LB_GAUSSIAN,
        ub=UB_GAUSSIAN,
        x_names=X_NAMES,
    )

    # run optimization and MCMC
    result = optimize.minimize(
        problem, progress_bar=False, n_starts=N_STARTS_SOME
    )
    result = sample.sample(
        problem,
        n_samples=N_SAMPLE_SOME,
        result=result,
    )

    # compute the log evidence using harmonic mean
    bridge_log_evidence = sample.evidence.bridge_sampling_log_evidence(result)
    laplace = sample.evidence.laplace_approximation_log_evidence(
        problem, result.optimize_result.x[0]
    )
    assert np.isclose(bridge_log_evidence, laplace, atol=tol)
