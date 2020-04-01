import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns

from edit_parallel_tempering import *

from AdaptiveMetropolis.adaptive_metropolis_sampler import AdaptiveMetropolisSampler

def p(x):
    return np.log(0.3*multivariate_normal.pdf(x, mean=-1.5, cov=0.1)+0.7*multivariate_normal.pdf(x, mean=2.5, cov=0.2))

sample = np.array([0.])
lower_bounds = np.array([-10.])
upper_bounds = np.array([10.])
covariance = np.identity(sample.size)

# could change n_samples to chain_length to avoid confusion
settings = {
    'n_samples': 1000,
    #Adaptive Metropolis settings
    'debug': False,
    'sample': sample,
    'covariance': covariance,
    'lower_bounds': lower_bounds,
    'upper_bounds': upper_bounds,
    'decay_constant': 0.51,
    'threshold_sample': 1,
    'regularization_factor': 1e-6,
    'log_posterior_callable': lambda x: p(x),
    #Parallel Tempering settings
    'sampler': AdaptiveMetropolisSampler,
    'n_temperatures': 5,
    'exp_temperature': 4,
    'temperature_nu': 1000,
    'temperature_eta': 100,
    'max_temp': 50000
}

AM_sampler = AdaptiveMetropolisSampler(settings)
resultAM = AM_sampler.sample()

resultPT = parallel_tempering(settings)

plt.figure()
plt.plot(range(settings['n_samples']), resultAM['samples'][0][0], 'ko', label='MCMC sample')
plt.xlabel('# iterations')
plt.ylabel('log10(x)')
plt.legend()
plt.savefig('edit_toyExample_x_AM.svg')
plt.close()

plt.figure()
plt.hist(resultAM['samples'][0][0])
plt.xlabel('log10(x)')
plt.ylabel('Occurrence')
plt.savefig('edit_toyExample_xHist_AM.svg')
plt.close()

plt.figure()
plt.plot(range(settings['n_samples']), resultPT['samples'][0][0], 'ko', label='MCMC sample')
plt.xlabel('# iterations')
plt.ylabel('log10(x)')
plt.legend()
plt.savefig('edit_toyExample_x_PT.svg')
plt.close()

plt.figure()
plt.hist(resultPT['samples'][0][0])
plt.xlabel('log10(x)')
plt.ylabel('Occurrence')
plt.savefig('edit_toyExample_xHist_PT.svg')
plt.close()
# print(resultPT['sample'].shape)

plt.figure(figsize=(5, 13))
sns.set_style('whitegrid')
for n_T in range(settings['n_temperatures']):
    plt.subplot(settings['n_temperatures'], 1, n_T+1)
    sns.kdeplot(resultPT['samples'][n_T][0], bw=0.1)
    # plt.hist(resultPT['sample'][n_T][0])
    plt.title('T'+str(n_T+1))
    plt.xlabel('x')
    plt.ylabel('Density')
plt.tight_layout()
plt.savefig('edit_toyExample_xKDE_PT.svg')
plt.close()

plt.figure()
sns.set_style('whitegrid')
sns.kdeplot(resultAM['samples'][0][0], bw=0.1)
# plt.hist(resultAM['sample'][0])
plt.xlabel('x')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig('edit_toyExample_xKDE_AM.svg')
plt.close()
