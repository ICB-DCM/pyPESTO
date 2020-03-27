import numpy as np
import matplotlib.pyplot as plt

from parallel_tempering import *
import adaptive_metropolis
# from sampling_fval import *
from scipy.integrate import solve_ivp


def model(k1, km1):
    return lambda t, y : np.array([km1*y[1] - k1*y[0], k1*y[0] - km1*y[1]])


def simulate(p):
    sol = solve_ivp(model(p[0], p[1]), (0, observable_timepoints[-1]), [1, 0],
                                 dense_output=True, rtol=1e-4, atol=1e-6).sol
    return sol(observable_timepoints)[1]


def objectiveFunction(theta_vec):
    simulation = simulate(np.power(10, theta_vec))
    sigma = 0.015
    diff = observable_measurements - simulation
    return -np.dot(diff, diff) / sigma**2


observable_measurements = np.array([0.0244, 0.0842, 0.1208,
                             0.1724, 0.2315, 0.2634,
                             0.2831, 0.3084, 0.3079,
                             0.3097, 0.3324])
observable_timepoints = np.arange(len(observable_measurements))
init = np.array([1, 0])

theta0 = np.array([-2., -3.])
# theta0 = np.log10(np.array([0.08594872, 0.1475647]))  # start at optimal parameters
theta_bounds_lower = np.array([-4, -4])
theta_bounds_upper = np.array([3, 3])
covariance0 = np.identity(theta0.size)

options = {
    'debug': True,
    'covariance': covariance0,
    'theta_bounds_lower': theta_bounds_lower,
    'theta_bounds_upper': theta_bounds_upper,
    'iterations': 500,
    'decay_rate': 0.5,
    'threshold_iteration': 5,
    'regularization_factor': 1e-6,
    'n_temperatures': 5,
    'exp_temperature': 4,
    'alpha': 0.5100,
    'temperature_nu': 1000,
    'memoryLength': 1,
    'temperature_eta': 100,
    'max_temp': 50000
}

# resultAM = adaptive_metropolis.adaptive_metropolis(
#     lambda t: objectiveFunction(t),
#     theta0, options)

resultAM = parallel_tempering(
    lambda t: objectiveFunction(t),
    theta0, options)

# print(np.median(resultAM['log_posterior']))

theta_true = [0.08594872, 0.1475647]
simulation_true = simulate(theta_true)
theta_S = np.power(10, np.median(resultAM['theta'][0], axis=1))
simulation_S = simulate(theta_S)

plt.figure()
plt.plot(observable_timepoints, simulation_true, 'k', label='Model simulation ($\Theta_{true}$)')
plt.plot(observable_timepoints, simulation_S, 'b', label='Model simulation ($\Theta_{MCMC}$)')
plt.plot(observable_timepoints, observable_measurements, 'or', label='Experimental data')
plt.xlabel('Time [au]')
plt.ylabel('B level [au]')
plt.legend()
plt.savefig('plotAM.svg')
plt.close()

plt.figure(figsize=(15, 5))
for n in range(2):
    plt.subplot(1, 2, n+1)
    plt.plot(range(options['iterations']), resultAM['theta'][0, n, :], 'o')
    plt.plot([0, options['iterations']], np.log10([theta_true[n], theta_true[n]]), 'k--')
    plt.xlabel('# iterations')
    plt.ylim([theta_bounds_lower[n],theta_bounds_upper[n]])
    plt.ylabel('log($\Theta_{'+str(n+1)+'}$)')
plt.savefig('testAM.svg')
plt.close()

import seaborn as sns
plt.figure(figsize=(10, 13))
sns.set_style('whitegrid')
counter = 1
for n_T in range(options['n_temperatures']):
    for n_par in [0, 1]:
        plt.subplot(options['n_temperatures'], 2, counter)
        plt.hist(resultAM['theta'][n_T,n_par,:],density=True,color='darkgrey')
        sns.kdeplot(resultAM['theta'][n_T,n_par,:], bw=0.5, color='darkviolet')
        plt.axvline(np.log10(theta_true[n_par]),color='k',lw=2,ls='--')
        plt.xlim([theta_bounds_lower[n_par], theta_bounds_upper[n_par]])
        plt.title('$T_'+str(n_T+1)+'$')
        plt.xlabel('log$_{10}(\Theta_{'+str(n_par+1)+'})$')
        plt.ylabel('Density')
        counter += 1
plt.tight_layout()
plt.savefig('CR_xKDE_PT.png')
plt.close()
