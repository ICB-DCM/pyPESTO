"""A set of helper functions"""
import numpy as np
import logging
from typing import Tuple

from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)


def calculate_ci_mcmc_sample(
        result: Result,
        ci_level: float = 0.95,
        exclude_burn_in: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray]:
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
) -> Tuple[np.ndarray, np.ndarray]:
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
    percentiles = 100 * np.array([(1-ci_level)/2, 1-(1-ci_level)/2])
    # Upper and lower bounds
    lb, ub = np.percentile(values, percentiles, **kwargs)
    return lb, ub
