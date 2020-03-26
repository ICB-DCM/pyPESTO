from typing import Callable
import numpy as np
from adaptive_metropolis import *


def parallel_tempering(
        log_posterior_callable: Callable,
        theta0: np.array,
        options: dict) -> dict:

    theta_bounds_lower = options['theta_bounds_lower']
    theta_bounds_upper = options['theta_bounds_upper']
    covariance0 = options['covariance']
    iterations = options['iterations']
    n_temperatures = options['n_temperatures']
    temperature_nu = options['temperature_nu']
    temperature_eta = options['temperature_eta']
    threshold_iteration = options['threshold_iteration']
    decay_rate = options['decay_rate']
    regularization_factor = options['regularization_factor']
    exp_temperature = options['exp_temperature']
    max_temp = options['max_temp']

    theta0 = theta0.astype(float)

    temperature = np.linspace(1, max_temp**(1/exp_temperature), n_temperatures)**exp_temperature
    beta = 1/temperature  # reciprocal temperature

    # Special case of AM: necessary due to linspace behavior
    if n_temperatures == 1:
        temperature = 1
        beta = 1

    if n_temperatures > 1:
        if covariance0.shape[0] == 1:
            covariance = np.tile(covariance0, (n_temperatures, 1, 1))
        elif covariance0.shape[0] == n_temperatures:
            covariance = covariance0
        elif len(covariance0.shape) < 3:
            covariance = np.tile(covariance0, (n_temperatures, 1, 1))
        else:
            raise ValueError("Dimension of options['covariance0'] is incorrect.")
        if theta0.shape[0] == 1:
            theta = np.tile(theta0, (n_temperatures, 1))
        elif theta0.shape[0] == n_temperatures:
            theta = theta0
        elif len(theta0.shape) < 3:
            theta = np.tile(theta0, (n_temperatures, 1))
        else:
            raise ValueError("Dimension of options['theta0'] is incorrect.")

    parameter_count = theta.shape[1]  # parameter number
    log_posterior_theta = np.empty([n_temperatures, 1])

    for n_T in range(n_temperatures):
        # Regularization covariance0
        covariance[n_T] = adaptive_metropolis_regularizer(covariance[n_T],
                                                          regularization_factor,
                                                          parameter_count,
                                                          MAGIC_DIVIDING_NUMBER=1000)
        # Evaluate posterior at theta0
        log_posterior_theta[n_T] = log_posterior_callable(theta[n_T])

    historical_mean = theta
    historical_covariance = covariance
    accepted_count = np.full([n_temperatures, iterations], 0)
    covariance_scaling_factor = np.full([n_temperatures, 1], 1)

    result = {'theta': np.full([n_temperatures, parameter_count, iterations], np.nan),
              'log_posterior': np.full([n_temperatures, iterations], np.nan)}

    acc_swap = np.full([1, n_temperatures-1], np.nan)
    prop_swap = np.full([1, n_temperatures-1], np.nan)
    a_bool = np.full([1, n_temperatures-1], np.nan)
    p_acc_swap = np.full([1, n_temperatures-1], np.nan)

    for i in range(iterations):
        # Do MCMC step for each temperature
        for n_T in range(n_temperatures):
            # Propose
            theta_update_result = adaptive_metropolis_update_theta(log_posterior_callable,
                                                                   theta[n_T],
                                                                   log_posterior_theta[n_T],
                                                                   covariance[n_T],
                                                                   theta_bounds_lower,
                                                                   theta_bounds_upper,
                                                                   False)
            if theta_update_result['accepted']:
                theta[n_T] = theta_update_result['theta']
                log_posterior_theta[n_T] = theta_update_result['log_posterior']
                accepted_count[n_T] += 1

        for n_T in range(n_temperatures):
            covariance_update_result = adaptive_metropolis_update_covariance(historical_mean[n_T],
                                                                   historical_covariance[n_T],
                                                                   theta[n_T],
                                                                   threshold_iteration,
                                                                   decay_rate,
                                                                   covariance_scaling_factor[n_T],
                                                                   theta_update_result['log_acceptance'],
                                                                   regularization_factor,
                                                                   parameter_count,
                                                                   i)
            historical_mean[n_T] = covariance_update_result['historical_mean']
            historical_covariance[n_T] = (
                covariance_update_result['historical_covariance'])
            covariance_scaling_factor[n_T] = (
                covariance_update_result['covariance_scaling_factor'])
            covariance[n_T] = covariance_update_result['covariance']

        # Swaps between all adjacent chains as in Vousden16
        if n_temperatures > 1:
            d_beta = beta[:-1] - beta[1:]
            for n_T in reversed(range(1, n_temperatures)):
                p_acc_swap[0, n_T - 1] = np.multiply(d_beta[n_T - 1],
                                                np.transpose(
                                                    log_posterior_theta[n_T] - log_posterior_theta[n_T - 1]
                                                ))
                a_bool[0, n_T - 1] = np.less(np.log(float(np.random.uniform(0, 1, 1))), p_acc_swap[0, n_T - 1])
                prop_swap[0, n_T - 1] = prop_swap[0, n_T - 1] + 1
                acc_swap[0, n_T - 1] = acc_swap[0, n_T - 1] + a_bool[0, n_T - 1]
                # As usually implemented when using PT
                if a_bool[0, n_T - 1]:
                    theta[[n_T, n_T - 1]] = theta[[n_T - 1, n_T]]
                    log_posterior_theta[[n_T, n_T - 1]] = log_posterior_theta[[n_T - 1, n_T]]
        # Adaptation of the temperature values (Vousden 2016)
        if n_temperatures > 1:
            kappa = temperature_nu / (i + 1 + temperature_nu) / temperature_eta
            ds = np.multiply(kappa, a_bool[0, :-1] - a_bool[0, 1:])
            dt = np.diff(1. / beta[:- 1])
            dt = np.multiply(dt, np.exp(ds))
            beta[:-1] = 1/np.cumsum(np.insert(dt, 0, 1))
        # Update results
        for n_T in range(n_temperatures):
            result['theta'][n_T, :, i] = theta[n_T]
            result['log_posterior'][n_T, i] = log_posterior_theta[n_T]

    return result
