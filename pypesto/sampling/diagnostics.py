from typing import Dict, List, Sequence, Union
import numpy as np

from ..result import Result
from .result import McmcPtResult
from .geweke_test import burnInBySequentialGeweke
from .auto_correlation import auto_correlation

def GewekeTest(result: Result, zscore: float = 2.):
    ''' Calculates the burn-in of MCMC chains.

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

    '''
    # get parameters and fval results as numpy arrays
    chain = np.array(result.sample_result['trace_x'][0])

    burn_in = burnInBySequentialGeweke(chain=chain, zscore=zscore)
    print('Geweke Burn-in index: '+str(burn_in))

    return burn_in

def ChainAutoCorrelation(result: Result, burn_in: int = 0):
    ''' Calculates the auto-correlation of the MCMC chains.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    burn_in:
        The burn in index obtained from convergence tests,
        e.g. Geweke. Default 0.

    Returns
    -------
    tau:
        Vector with estimated auto-correlation values.

    '''
    # get parameters and fval results as numpy arrays
    # discarding warm up phase
    chain = np.array(result.sample_result['trace_x'][0][burn_in:,:])

    tau = auto_correlation(chain=chain)

    return tau
