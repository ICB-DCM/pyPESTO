from typing import Dict
import numpy as np
from AdaptiveMetropolis.adaptive_metropolis_sampler_parent import Sampler
from parallel_tempering_methods import sample_parallel_tempering

def parallel_tempering(
        settings: Dict) -> Dict:
    '''

    Parameters
    ----------
    settings

    Returns
    -------

    '''
    # Initialize variable
    samplers = []
    n_temperatures = settings['n_temperatures']

    # Loop over temperatures
    for n_T in range(n_temperatures):
        # Initialize sampling history
        samplers.append(settings['sampler'](settings=settings))

    # Perform parallel tempering
    result = sample_parallel_tempering(samplers,
                                       settings)
    return result
