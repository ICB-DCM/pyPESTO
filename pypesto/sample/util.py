import numpy as np
import logging
from typing import Tuple

from ..result import Result
from .diagnostics import geweke_test

logger = logging.getLogger(__name__)


def calculate_samples_ci(
        result: Result, alpha: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate parameter confidence intervals based on MCMC samples.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    alpha:
        Lower tail probability, defaults to 95% interval.

    Returns
    -------
    lb, ub:
        Bounds of the MCMC percentile-based confidence interval.
    """
    # Check if burn in index is available
    if result.sample_result.burn_in is None:
        geweke_test(result)

    # Get burn in index
    burn_in = result.sample_result.burn_in

    # Get converged parameter samples as numpy arrays
    chain = np.asarray(result.sample_result.trace_x[0, burn_in:, :])

    # Get percentile values corresponding to alpha
    percentiles = 100 * np.array([(1-alpha)/2, 1-(1-alpha)/2])

    # Get samples' upper and lower bounds
    lb, ub = np.percentile(chain, percentiles, axis=0)

    return lb, ub
