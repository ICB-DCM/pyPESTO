"""A set of helper functions."""

import logging
import os
from typing import Optional

import numpy as np

from ..C import PYPESTO_MAX_N_SAMPLES
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
    trace_neglogpost: np.ndarray,
    trace_neglogprior: np.ndarray,
    prior_samples: Optional[np.ndarray] = None,
    neg_log_likelihood_fun: Optional[callable] = None,
) -> float:
    """
    Compute the log evidence using the harmonic mean estimator. If samples from the prior are provided, the stabilized harmonic mean is computed (recommended).

    Parameters
    ----------
    trace_neglogpost: np.ndarray
        Negative log posterior samples.
    trace_neglogprior: np.ndarray
        Negative log prior samples.
    prior_samples: np.ndarray (n_samples, n_parameters)
        Samples from the prior distribution.
    neg_log_likelihood_fun: callable
        Function to evaluate the negative log likelihood.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp

    # compute negative log likelihoods from traces
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
