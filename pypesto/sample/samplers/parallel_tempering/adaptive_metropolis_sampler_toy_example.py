import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

### Elba's toy problem
def p(x):
    return np.log(0.3*multivariate_normal.pdf(x, mean=-1.5, cov=0.1)+0.7*multivariate_normal.pdf(x, mean=2.5, cov=0.2))

sample = np.array([0.])
lower_bounds = np.array([-10])
upper_bounds = np.array([10])
covariance = np.identity(sample.size)
###

settings = {
    'log_posterior_callable': lambda x: p(x),
    'debug': True,
    'covariance': covariance,
    'lower_bounds': lower_bounds,
    'upper_bounds': upper_bounds,
    'decay_constant': 0.5,
    'threshold_sample': 1,
    'regularization_factor': 1e-6,
    'n_samples': 1000,
    'sample': sample,
}

# Setup "burn-in"
from adaptive_metropolis_sampler import AdaptiveMetropolisSampler
sampler = AdaptiveMetropolisSampler(settings=settings)
print('first 1000')
chains = sampler.sample()
# Save "burn-in"
sampler.save_state('checkpoint.pickle')
print(sampler.get_state('chain')['samples_log_posterior'])

# Generate 100 samples with "burn-in"
# These 100 samples will be in addition to the "burn-in" samples
# Hence, the chain will contain 200 samples
print('first a further 1000, for a total of 2000 samples')
chains = sampler.sample(1000)
print(sampler.get_state('chain')['samples_log_posterior'])

# Generate another 100 samples with "burn-in", by reading
# in the saved file. Again, the chain will contain 200 samples.
sampler.load_state('checkpoint.pickle')
print('reload first 1000 samples, find another 1000 samples, for a total of 2000')
chains = sampler.sample(1000)
print(sampler.get_state('chain')['samples_log_posterior'])

print(chains)

# Elba's toy problem plotting code (slightly modified)
plt.figure()
plt.plot(range(sampler.state.n_samples), chains[0]['samples'][0], 'ko', label='MCMC sample')
plt.xlabel('# iterations')
plt.ylabel('log10(x)')
plt.legend()
plt.show()
plt.close()

plt.figure()
plt.hist(chains[0]['samples'][0])
plt.xlabel('log10(x)')
plt.ylabel('Occurrence')
plt.show()
plt.close()

plt.figure()
plt.hist(chains[0]['samples_log_posterior'])
plt.xlabel('log posterior')
plt.ylabel('Occurrence')
plt.show()
plt.close()
