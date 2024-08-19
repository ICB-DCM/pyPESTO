"""A set of helper functions."""

import logging
import os
from typing import Optional

import numpy as np

from ..C import PYPESTO_MAX_N_SAMPLES
from ..objective import (
    AggregatedObjective,
    NegLogParameterPriors,
    NegLogPriors,
)
from ..optimize.util import laplace_approximation_log_evidence
from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)


def calculate_ci_mcmc_sample(
    result: Result,
    ci_level: float = 0.95,
    exclude_burn_in: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate parameter credibility intervals based on MCMC samples.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    ci_level:
        Lower tail probability, defaults to 95% interval.

    Returns
    -------
    lb, ub:
        Bounds of the MCMC percentile-based confidence interval.
    """
    burn_in = 0
    if exclude_burn_in:
        # Check if burn in index is available
        if result.sample_result.burn_in is None:
            geweke_test(result)

        # Get burn in index
        burn_in = result.sample_result.burn_in

    # Get converged parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    lb, ub = calculate_ci(chain, ci_level=ci_level, axis=0)
    return lb, ub


def calculate_ci_mcmc_sample_prediction(
    simulated_values: np.ndarray,
    ci_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate prediction credibility intervals based on MCMC samples.

    Parameters
    ----------
    simulated_values:
        Simulated model states or model observables.
    ci_level:
        Lower tail probability, defaults to 95% interval.

    Returns
    -------
    lb, ub:
        Bounds of the MCMC-based prediction confidence interval.
    """
    lb, ub = calculate_ci(simulated_values, ci_level=ci_level, axis=1)
    return lb, ub


def calculate_ci(
    values: np.ndarray,
    ci_level: float,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate confidence/credibility levels using percentiles.

    Parameters
    ----------
    values:
        The values used to calculate percentiles.
    ci_level:
        Lower tail probability.
    kwargs:
        Additional keyword arguments are passed to the `numpy.percentile` call.

    Returns
    -------
    lb, ub:
        Bounds of the confidence/credibility interval.
    """
    # Percentile values corresponding to the CI level
    percentiles = 100 * np.array([(1 - ci_level) / 2, 1 - (1 - ci_level) / 2])
    # Upper and lower bounds
    lb, ub = np.percentile(values, percentiles, **kwargs)
    return lb, ub


def bound_n_samples_from_env(n_samples: int):
    """Bound number of samples from environment variable.

    Uses environment variable `PYPESTO_MAX_N_SAMPLES`.
    This is used to speed up testing, while in application it should not
    be used.

    Parameters
    ----------
    n_samples: Number of samples desired.

    Returns
    -------
    n_samples_new:
        The original number of samples, or the minimum with the environment
        variable, if exists.
    """
    if PYPESTO_MAX_N_SAMPLES not in os.environ:
        return n_samples
    n_samples_new = min(n_samples, int(os.environ[PYPESTO_MAX_N_SAMPLES]))

    logger.info(
        f"Bounding number of samples from {n_samples} to {n_samples_new} via "
        f"environment variable {PYPESTO_MAX_N_SAMPLES}"
    )

    return n_samples_new


def harmonic_mean_log_evidence(
    result: Result,
    prior_samples: Optional[np.ndarray] = None,
    neg_log_likelihood_fun: Optional[callable] = None,
) -> float:
    """
    Compute the log evidence using the harmonic mean estimator.

    Stabilized harmonic mean estimator is used if prior samples are provided.
    Newton and Raftery (1994): https://doi.org/10.1111/j.2517-6161.1994.tb01956.x

    Parameters
    ----------
    result: Result
    prior_samples: np.ndarray (n_samples, n_parameters)
        Samples from the prior distribution. If samples from the prior are provided,
        the stabilized harmonic mean is computed (recommended). Then, the likelihood function must be provided as well.
    neg_log_likelihood_fun: callable
        Function to evaluate the negative log likelihood.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    # compute negative log likelihood from traces
    burn_in = geweke_test(result)
    trace_neglogpost = result.sample_result.trace_neglogpost[0, burn_in:]
    trace_neglogprior = result.sample_result.trace_neglogprior[0, burn_in:]
    neg_log_likelihoods_posterior = trace_neglogpost - trace_neglogprior

    if prior_samples is None:
        # compute harmonic mean from samples
        return -logsumexp(neg_log_likelihoods_posterior) + np.log(
            neg_log_likelihoods_posterior.size
        )

    # compute stabilized harmonic mean
    if prior_samples is not None and neg_log_likelihood_fun is None:
        raise ValueError(
            "you need to provide a likelihood function to evaluate prior samples"
        )

    # compute delta (ratio of prior to posterior samples)
    n_samples_prior = len(prior_samples)
    n_samples_posterior = len(trace_neglogpost)
    delta = n_samples_prior / (n_samples_prior + n_samples_posterior)
    neg_log_likelihoods_prior = np.array(
        [neg_log_likelihood_fun(x) for x in prior_samples]
    )
    log_likelihoods_stack = -np.concatenate(
        [neg_log_likelihoods_prior, neg_log_likelihoods_posterior]
    )

    def _log_evidence_objective(log_p: float):
        # Helper function to compute the log evidence with stabilized harmonic mean
        log_w_i = logsumexp(
            np.stack(
                (
                    log_p * np.ones_like(log_likelihoods_stack),
                    log_likelihoods_stack,
                ),
                axis=1,
            ),
            b=np.array([delta, 1 - delta]),
            axis=1,
        )
        res, sign = logsumexp(
            [
                log_p,
                logsumexp(log_likelihoods_stack - log_w_i)
                - logsumexp(-log_w_i),
            ],
            b=[1, -1],
            return_sign=True,
        )
        return sign * res

    sol = minimize_scalar(_log_evidence_objective)
    return sol.x


def bridge_sampling(
    result: Result,
    n_posterior_samples_init: Optional[int] = None,
    initial_guess_log_evidence: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> float:
    """
    Compute the log evidence using bridge sampling.

    Based on "A Tutorial on Bridge Sampling" by Gronau et al. (2017): https://api.semanticscholar.org/CorpusID:5447695.
    Using the optimal bridge function by Meng and Wong (1996) which minimises the relative mean-squared error.
    Proposal function is calibrated using posterior samples, which are not used for the final bridge estimate
    (as this may result in an underestimation of the marginal likelihood, see Overstall and Forster (2010)).

    Parameters
    ----------
    result: Result
        The pyPESTO result object with filled sample result.
    n_posterior_samples_init: int
        Number of samples used to calibrate the proposal function. By default, half of the posterior samples are used.
    initial_guess_log_evidence: np.ndarray
        Initial guess for the log evidence. By default, the Laplace approximation is used to compute the initial guess.
    max_iter: int
        Maximum number of iterations. Default is 1000.
    tol: float
        Tolerance for convergence. Default is 1e-6.
    """
    from scipy import stats
    from scipy.special import logsumexp

    if result.sample_result is None:
        raise ValueError("No samples available. Run sampling first.")
    if not isinstance(result.problem.objective, AggregatedObjective):
        raise ValueError("Objective must be an AggregatedObjective.")

    # use Laplace approximation to get initial guess for p(y)
    if initial_guess_log_evidence is None:
        initial_guess_log_evidence = laplace_approximation_log_evidence(
            problem=result.problem, x=result.optimize_result.x[0]
        )
    # extract posterior samples
    burn_in = geweke_test(result)
    posterior_samples = result.sample_result.trace_x[0, burn_in:]

    # build proposal function from posterior samples
    if n_posterior_samples_init is None:
        n_posterior_samples_init = int(posterior_samples.shape[0] * 0.5)
    # randomly select samples for calibration
    calibration_index = np.random.choice(
        np.arange(posterior_samples.shape[0]),
        n_posterior_samples_init,
        replace=False,
    )
    samples_calibration = posterior_samples[calibration_index]
    # remove calibration samples from posterior samples
    posterior_samples = posterior_samples[
        [
            j
            for j in range(posterior_samples.shape[0])
            if j not in calibration_index
        ]
    ]
    # generate proposal samples and define proposal function
    n_proposal_samples = posterior_samples.shape[0]
    posterior_mean = np.mean(samples_calibration, axis=0)
    posterior_cov = np.cov(samples_calibration.T)
    if posterior_cov.size == 1:
        # univariate case
        proposal_samples = np.random.normal(
            loc=posterior_mean,
            scale=np.sqrt(posterior_cov),
            size=n_proposal_samples,
        )
        proposal_samples = proposal_samples.reshape(-1, 1)
    else:
        # multivariate case
        proposal_samples = np.random.multivariate_normal(
            mean=posterior_mean, cov=posterior_cov, size=n_proposal_samples
        )
    log_proposal_fun = stats.multivariate_normal(
        mean=posterior_mean, cov=posterior_cov
    ).logpdf

    # Compute the weights for the bridge sampling estimate
    log_s1 = np.log(
        posterior_samples.shape[0]
        / (posterior_samples.shape[0] + n_proposal_samples)
    )
    log_s2 = np.log(
        n_proposal_samples / (posterior_samples.shape[0] + n_proposal_samples)
    )

    # Start with the initial guess for p(y)
    log_p_y = initial_guess_log_evidence

    # Compute the log-likelihood, log-prior, and log-proposal for the posterior and proposal samples
    # assumes that the objective function is the negative log-likelihood + negative log-prior

    # get index of prior in the objective function
    likelihood_fun_indices = []
    for i, obj in enumerate(result.problem.objective._objectives):
        if not isinstance(obj, NegLogParameterPriors) and not isinstance(
            obj, NegLogPriors
        ):
            likelihood_fun_indices.append(i)

    def log_likelihood_fun(x_array):
        return np.array(
            [
                np.sum(
                    [
                        -obj(
                            result.problem.get_full_vector(
                                x=x, x_fixed_vals=result.problem.x_fixed_vals
                            )
                        )
                        for obj_i, obj in enumerate(
                            result.problem.objective._objectives
                        )
                        if obj_i in likelihood_fun_indices
                    ]
                )
                for x in x_array
            ]
        )

    def log_prior_fun(x_array):
        return np.array(
            [
                np.sum(
                    [
                        -obj(
                            result.problem.get_full_vector(
                                x=x, x_fixed_vals=result.problem.x_fixed_vals
                            )
                        )
                        for obj_i, obj in enumerate(
                            result.problem.objective._objectives
                        )
                        if obj_i not in likelihood_fun_indices
                    ]
                )
                for x in x_array
            ]
        )

    log_likelihood_posterior = log_likelihood_fun(posterior_samples)
    log_prior_posterior = log_prior_fun(posterior_samples)
    log_proposal_posterior = log_proposal_fun(posterior_samples)

    log_likelihood_proposal = log_likelihood_fun(proposal_samples)
    log_prior_proposal = log_prior_fun(proposal_samples)
    log_proposal_proposal = log_proposal_fun(proposal_samples)

    log_h_posterior_1 = log_s1 + log_likelihood_posterior + log_prior_posterior
    log_h_proposal_1 = log_s1 + log_likelihood_proposal + log_prior_proposal
    for i in range(max_iter):
        # Compute h(θ) for posterior samples
        log_h_posterior_2 = log_s2 + log_p_y + log_proposal_posterior
        log_h_posterior = logsumexp([log_h_posterior_1, log_h_posterior_2])

        # Compute h(θ) for proposal samples
        log_h_proposal_2 = log_s2 + log_p_y + log_proposal_proposal
        log_h_proposal = logsumexp([log_h_proposal_1, log_h_proposal_2])

        # Calculate the numerator and denominator for the bridge sampling estimate
        temp = log_likelihood_proposal + log_prior_proposal + log_h_proposal
        log_numerator = logsumexp(temp) - np.log(
            temp.size
        )  # compute mean in log space
        temp = log_proposal_posterior + log_h_posterior
        log_denominator = logsumexp(temp) - np.log(
            temp.size
        )  # compute mean in log space

        # Update p(y)
        log_p_y_new = log_numerator - log_denominator

        # Check for convergence
        if abs(log_p_y_new - log_p_y) < tol:
            break

        log_p_y = log_p_y_new

        if i == max_iter - 1:
            logger.warning(
                "Bridge sampling did not converge in the given number of iterations."
            )

    return log_p_y
