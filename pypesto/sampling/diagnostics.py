import numpy as np

from ..result import Result
from .geweke_test import burn_in_by_sequential_geweke


def geweke_test(result: Result,
                zscore: float = 2.):
    """
    Calculates the burn-in of MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    zscore:
        The Geweke test threshold. Default 2.

    Returns
    -------
    burn_in:
        Iteration where the first and the last fraction of the chain
        do not differ significantly regarding Geweke test -> Burn-In

    """
    # Get parameter samples as numpy arrays
    chain = np.array(result.sample_result.trace_x[0])

    # Calculate burn in index
    burn_in = burn_in_by_sequential_geweke(chain=chain,
                                           zscore=zscore)
    print('Geweke Burn-in index: '+str(burn_in))

    result.sample_result.burn_in = burn_in

    return result
