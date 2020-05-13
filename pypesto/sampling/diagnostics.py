from typing import Dict, List, Sequence, Union
import numpy as np

from ..result import Result
from .result import McmcPtResult
from .geweke_test import burnInBySequentialGeweke

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
