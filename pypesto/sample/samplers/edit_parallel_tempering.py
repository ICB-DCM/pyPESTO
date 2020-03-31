import numpy as np
from AdaptiveMetropolis.adaptive_metropolis_sampler_parent import Sampler

def parallel_tempering(settings: dict) -> dict:
    n_samples = settings['n_samples']
    n_temperatures = settings['n_temperatures']
    temperature_nu = settings['temperature_nu']
    temperature_eta = settings['temperature_eta']
    exp_temperature = settings['exp_temperature']
    max_temp = settings['max_temp']

    temperature = np.linspace(1, max_temp**(1/exp_temperature), n_temperatures)**exp_temperature
    beta = 1/temperature  # reciprocal temperature

    # Special case of AM: necessary due to linspace behavior
    if n_temperatures == 1:
        temperature = 1
        beta = 1

    n_parameters = len(settings['sample'])  # parameter number

    samplers = []
    for n_T in range(n_temperatures):
        samplers.append(settings['sampler'](settings=settings))

    result = {'samples': np.full([n_temperatures, n_parameters, n_samples], np.nan),
              'samples_log_posterior': np.full([n_temperatures, n_samples], np.nan)}

    acc_swap = np.full([1, n_temperatures-1], np.nan)
    prop_swap = np.full([1, n_temperatures-1], np.nan)
    a_bool = np.full([1, n_temperatures-1], np.nan)
    p_acc_swap = np.full([1, n_temperatures-1], np.nan)

    for i in range(n_samples):
        # Do MCMC step for each temperature
        for n_T in range(n_temperatures):
            samplers[n_T].sample(1)

        # Swaps between all adjacent chains as in Vousden16
        if n_temperatures > 1:
            d_beta = beta[:-1] - beta[1:]
            for n_T in reversed(range(1, n_temperatures)):
                p_acc_swap[0, n_T - 1] = np.multiply(d_beta[n_T - 1],
                                                np.transpose(
                                                    samplers[n_T].get_last_sample('samples_log_posterior') - samplers[n_T - 1].get_last_sample('samples_log_posterior')
                                                ))
                a_bool[0, n_T - 1] = np.less(np.log(float(np.random.uniform(0, 1, 1))), p_acc_swap[0, n_T - 1])
                prop_swap[0, n_T - 1] = prop_swap[0, n_T - 1] + 1
                acc_swap[0, n_T - 1] = acc_swap[0, n_T - 1] + a_bool[0, n_T - 1]
                # As usually implemented when using PT
                if a_bool[0, n_T - 1]:
                    samplers[n_T - 1], samplers[n_T] = Sampler.swap_last_samples(samplers[n_T], samplers[n_T - 1])

        # Adaptation of the temperature values (Vousden 2016)
        if n_temperatures > 1:
            kappa = temperature_nu / (i + 1 + temperature_nu) / temperature_eta
            ds = np.multiply(kappa, a_bool[0, :-1] - a_bool[0, 1:])
            dt = np.diff(1. / beta[:- 1])
            dt = np.multiply(dt, np.exp(ds))
            beta[:-1] = 1/np.cumsum(np.insert(dt, 0, 1))

        # Update results
        for n_T in range(n_temperatures):
            result['samples'][n_T, :, i] = samplers[n_T].get_last_sample('samples')
            result['samples_log_posterior'][n_T, i] = samplers[n_T].get_last_sample('samples_log_posterior')

    return result
