"""Calculate different diagnostics of the sampling result."""

import logging

import numpy as np

from ..result import Result
from .auto_correlation import autocorrelation_sokal
from .geweke_test import burn_in_by_sequential_geweke

logger = logging.getLogger(__name__)


def geweke_test(
    result: Result, zscore: float = 2.0, chain_number: int = 0
) -> int:
    """
    Calculate the burn-in of MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    zscore:
        The Geweke test threshold.
    chain_number:
        The chain number to be used for the Geweke test (in a parallel tempering setting).
        Usually we are only interested in the first chain.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction of the chain
        do not differ significantly regarding Geweke test -> Burn-In
    """
    if chain_number == 0:
        # test if burn-in is already calculated
        if result.sample_result.burn_in is not None:
            logger.info(
                f"Burn-in index ({result.sample_result.burn_in}) already calculated. Skipping Geweke test."
            )
            return result.sample_result.burn_in

    # Get parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[chain_number])

    # Calculate burn in index
    burn_in = burn_in_by_sequential_geweke(chain=chain, zscore=zscore)

    if chain_number == 0:
        # Log
        logger.info(f"Geweke burn-in index: {burn_in}")

        # Fill in burn-in value into result
        result.sample_result.burn_in = burn_in

    return burn_in


def auto_correlation(result: Result) -> float:
    """
    Calculate the autocorrelation of the MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.

    Returns
    -------
    auto_correlation:
        Estimate of the integrated autocorrelation time of
        the MCMC chains.
    """
    # Check if burn in index is available
    if result.sample_result.burn_in is None:
        geweke_test(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get chain length
    chain_length = result.sample_result.trace_x.shape[1]

    if burn_in == chain_length:
        logger.warning(
            "The autocorrelation can not "
            "be estimated. The chain seems to "
            "not have converged yet.\n"
            "You may want to use a larger number "
            "of samples."
        )
        return None

    # Get converged parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    # Calculate autocorrelation
    auto_correlation_vector = autocorrelation_sokal(chain=chain)

    # Take the maximum over all components
    _auto_correlation = max(auto_correlation_vector)

    # Log
    logger.info(f"Estimated chain autocorrelation: {_auto_correlation}")

    # Fill in autocorrelation value into result
    result.sample_result.auto_correlation = _auto_correlation

    return _auto_correlation


def effective_sample_size(result: Result) -> float:
    """
    Calculate the effective sample size of the MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.

    Returns
    -------
    ess:
        Estimate of the effective sample size of
        the MCMC chains.
    """
    # Check if autocorrelation is available
    if result.sample_result.auto_correlation is None:
        # Calculate autocorrelation
        auto_correlation(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get estimated chain autocorrelation
    _auto_correlation = result.sample_result.auto_correlation

    if _auto_correlation is None:
        return None

    # Get converged parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    # Get length of the converged chain
    N = chain.shape[0]

    # Calculate effective sample size
    ess = N / (1 + _auto_correlation)

    # Log
    logger.info(f"Estimated effective sample size: {ess}")

    # Fill in effective sample size value into result
    result.sample_result.effective_sample_size = ess

    return ess
