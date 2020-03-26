import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from parallel_tempering import *
import adaptive_metropolis
# from sampling_fval import *
from scipy.integrate import solve_ivp


def p(x):
    return np.log(0.3*multivariate_normal.pdf(x, mean=-1.5, cov=0.1)+0.7*multivariate_normal.pdf(x, mean=2.5, cov=0.2))


theta0 = np.array([0.])
# theta0 = np.log10(np.array([0.08594872, 0.1475647]))  # start at optimal parameters
theta_bounds_lower = np.array([-10])
theta_bounds_upper = np.array([10])
covariance0 = np.identity(theta0.size)

options = {
    'debug': True,
    'covariance': covariance0,
    'theta_bounds_lower': theta_bounds_lower,
    'theta_bounds_upper': theta_bounds_upper,
    'iterations': 10000,
    'decay_rate': 0.51,
    'threshold_iteration': 1,
    'regularization_factor': 1e-6,
    'n_temperatures': 2,
    'exp_temperature': 4,
    'alpha': 0.5100,
    'temperature_nu': 1000,
    'memoryLength': 1,
    'temperature_eta': 100,
    'max_temp': 50000
}

resultAM = adaptive_metropolis.adaptive_metropolis(
    lambda x: p(x), theta0, options)

# print(np.median(resultAM['log_posterior']))
# print(resultAM['theta'])

plt.figure()
plt.plot(range(options['iterations']), resultAM['theta'][0], 'ko', label='MCMC sample')
plt.xlabel('# iterations')
plt.ylabel('log10(x)')
plt.legend()
plt.savefig('toyExample_x_AM.svg')
plt.close()

plt.figure()
plt.hist(resultAM['theta'][0])
plt.xlabel('log10(x)')
plt.ylabel('Occurrence')
plt.savefig('toyExample_xHist_AM.svg')
plt.close()

plt.figure()
plt.hist(resultAM['log_posterior'])
plt.xlabel('log posterior')
plt.ylabel('Occurrence')
plt.savefig('toyExample_logPost_AM.svg')
plt.close()
