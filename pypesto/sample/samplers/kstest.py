import numpy as np
from scipy.stats import kstest, norm

from parallel_tempering import parallel_tempering
from AdaptiveMetropolis.adaptive_metropolis_sampler import AdaptiveMetropolisSampler

def logpdf(x):
    return norm.logpdf(x)

sample = np.array([0])
lower_bounds = np.array([-10])
upper_bounds = np.array([10])
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
    'log_posterior_callable': logpdf,
    #Parallel Tempering settings
    'sampler': AdaptiveMetropolisSampler,
    'n_temperatures': 5,
    'exp_temperature': 4,
    'temperature_nu': 1000,
    'temperature_eta': 100,
    'max_temp': 50000
}

samplerAM = AdaptiveMetropolisSampler(settings)
resultAM = samplerAM.sample()
kstestAM = kstest(resultAM['samples'][0][0], 'norm')
print('Kolmogorov-Smirnov test for the Adaptive Metropolis sampler.\nStatistic: '
        f'{kstestAM[0]:.5f}'
      '\np-value: '
      f'{kstestAM[1]:.15f}')

resultPT = parallel_tempering(settings)
kstestPT = kstest(resultPT['samples'][0][0], 'norm')
print('Kolmogorov-Smirnov test for Parallel Tempering with the Adaptive Metropolis sampler.\nStatistic: '
        f'{kstestPT[0]:.5f}'
      '\np-value: '
      f'{kstestPT[1]:.15f}')
