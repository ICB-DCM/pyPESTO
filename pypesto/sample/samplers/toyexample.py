import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

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
    'iterations': int(1e3),
    'decay_rate': 0.51,
    'threshold_iteration': 1,
    'regularization_factor': 1e-6,
    'n_temperatures': 5,
    'exp_temperature': 4,
    'alpha': 0.5100,
    'temperature_nu': 1000,
    'memoryLength': 1,
    'temperature_eta': 100,
    'max_temp': 50000
}

np.random.seed(0)
resultAM = adaptive_metropolis.adaptive_metropolis(
    lambda x: p(x), theta0, options)

np.random.seed(0)
resultPT = parallel_tempering(
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
plt.plot(range(options['iterations']), resultPT['theta'][0][0], 'ko', label='MCMC sample')
plt.xlabel('# iterations')
plt.ylabel('log10(x)')
plt.legend()
plt.savefig('toyExample_x_PT.svg')
plt.close()

plt.figure()
plt.hist(resultPT['theta'][0][0])
plt.xlabel('log10(x)')
plt.ylabel('Occurrence')
plt.savefig('toyExample_xHist_PT.svg')
plt.close()
# print(resultPT['theta'].shape)

plt.figure(figsize=(5, 13))
sns.set_style('whitegrid')
for n_T in range(options['n_temperatures']):
    plt.subplot(options['n_temperatures'], 1, n_T+1)
    sns.kdeplot(resultPT['theta'][n_T][0], bw=0.5)
    plt.title('T = '+str(n_T+1))
    plt.xlabel('x')
    plt.ylabel('Density')
plt.tight_layout()
plt.savefig('toyExample_xKDE_PT.svg')
plt.close()

plt.figure()
sns.set_style('whitegrid')
sns.kdeplot(resultAM['theta'][0], bw=0.5)
plt.xlabel('x')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('toyExample_xKDE_AM.svg')
plt.close()
