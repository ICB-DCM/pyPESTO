from typing import Callable, Tuple, Sequence, Dict
from ..sampler import Sampler
import numpy as np
import copy


def initialize_reciprocal_temperature(
        settings: Dict
) -> np.ndarray:
    '''

    Parameters
    ----------
    settings

    Returns
    -------

    '''
    # Initialize variables
    n_temperatures = settings['n_temperatures']
    exp_temperature = settings['exp_temperature']
    max_temp = settings['max_temp']

    # Calculate temperature values
    temperature = np.linspace(1,
                              max_temp ** (1 / exp_temperature),
                              n_temperatures) ** exp_temperature
    # Calculate reciprocal temperature
    beta = 1 / temperature

    # Special case of AM: necessary due to linspace behavior
    if n_temperatures == 1:
        beta = 1

    return beta


def swapping_adjacent_chains(
        samplers: list,
        tempering_variables: Dict
) -> Tuple[list, Dict]:
    '''

    Parameters
    ----------
    samplers
    tempering_variables

    Returns
    -------

    '''
    # Initialize variables
    beta = tempering_variables['beta']
    acc_swap = tempering_variables['acc_swap']
    prop_swap = tempering_variables['prop_swap']
    a_bool = tempering_variables['a_bool']
    p_acc_swap = tempering_variables['p_acc_swap']

    # Number of parameters
    n_temperatures = len(beta)

    d_beta = beta[:-1] - beta[1:]
    for n_T in reversed(range(1, n_temperatures)):
        p_acc_swap[0, n_T - 1] = np.multiply(d_beta[n_T - 1],
                                             np.transpose(
                                                 samplers[n_T].get_last_sample('samples_log_posterior') - samplers[
                                                     n_T - 1].get_last_sample('samples_log_posterior')
                                             ))
        a_bool[0, n_T - 1] = np.less(np.log(float(np.random.uniform(0, 1, 1))),
                                     p_acc_swap[0, n_T - 1]
                                     )
        prop_swap[0, n_T - 1] = prop_swap[0, n_T - 1] + 1
        acc_swap[0, n_T - 1] = acc_swap[0, n_T - 1] + a_bool[0, n_T - 1]
        # As usually implemented when using PT
        if a_bool[0, n_T - 1]:
            samplers[n_T - 1], samplers[n_T] = Sampler.swap_last_samples(samplers[n_T],
                                                                         samplers[n_T - 1]
                                                                         )

        return (samplers, tempering_variables)


def adaptation_temperature_values(
        iteration: int,
        temperature_nu: float,
        temperature_eta: float,
        tempering_variables: Dict
) -> Dict:
    '''

    Parameters
    ----------
    iteration
    temperature_nu
    temperature_eta
    tempering_variables

    Returns
    -------

    '''
    # Initialize variables
    beta = tempering_variables['beta']
    a_bool = tempering_variables['a_bool']

    # Adaptation of temperature values
    kappa = temperature_nu / (iteration + 1 + temperature_nu) / temperature_eta
    ds = np.multiply(kappa, a_bool[0, :-1] - a_bool[0, 1:])
    dt = np.diff(1. / beta[:- 1])
    dt = np.multiply(dt, np.exp(ds))
    beta[:-1] = 1 / np.cumsum(np.insert(dt, 0, 1))

    return tempering_variables


def swapping_and_adaptation(
        samplers: list,
        tempering_variables: Dict,
        iteration: int,
        settings: Dict
) -> Tuple[list, Dict]:
    '''

    Parameters
    ----------
    samplers
    tempering_variables
    iteration
    settings

    Returns
    -------

    '''
    # Initialize variables
    temperature_nu = settings['temperature_nu']
    temperature_eta = settings['temperature_eta']

    # Swaps between all adjacent chains as in Vousden16
    samplers, tempering_variables = swapping_adjacent_chains(samplers,
                                                             tempering_variables)
    # Adaptation of the temperature values (Vousden 2016)
    tempering_variables = adaptation_temperature_values(iteration,
                                                        temperature_nu,
                                                        temperature_eta,
                                                        tempering_variables)
    return  (samplers, tempering_variables)


def sample_parallel_tempering(
        samplers: list,
        settings: Dict
) -> Dict:
    '''

    Parameters
    ----------
    samplers
    settings

    Returns
    -------

    '''
    # Initialize variables
    n_samples = settings['n_samples']
    n_temperatures = settings['n_temperatures']
    n_parameters = len(settings['sample'])  # parameter number
    exp_temperature = settings['exp_temperature']
    max_temp = settings['max_temp']

    # Initialize reciprocal temperature (1/temperature)
    beta = initialize_reciprocal_temperature(settings)

    # Initialize tempering variables
    tempering_variables = {
        'beta': beta,
        'acc_swap': np.full([1, n_temperatures - 1], np.nan),
        'prop_swap': np.full([1, n_temperatures - 1], np.nan),
        'a_bool': np.full([1, n_temperatures - 1], np.nan),
        'p_acc_swap': np.full([1, n_temperatures - 1], np.nan)}

    # Initialize result object
    result = {
        'samples': np.full([n_temperatures,
                                  n_parameters,
                                  n_samples],
                                 np.nan),
        'samples_log_posterior': np.full([n_temperatures,
                                          n_samples],
                                          np.nan)}

    # Loop over total number of samples
    for i in range(n_samples):
        # Do MCMC step for each temperature
        for n_T in range(n_temperatures):
            samplers[n_T].sample(1)

        # Swaps between all adjacent chains
        # and adaptation of the temperature values
        if n_temperatures > 1:
            swapping_and_adaptation(samplers,
                                    tempering_variables,
                                    i,
                                    settings)
        # Update results
        for n_T in range(n_temperatures):
            result['samples'][n_T, :, i] = samplers[n_T].get_last_sample('samples')
            result['samples_log_posterior'][n_T, i] = samplers[n_T].get_last_sample('samples_log_posterior')

    return result
