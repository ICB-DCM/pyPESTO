"""
This is for testing optimization of the pypesto.Objective.
"""


import numpy as np
from scipy.stats import kstest, norm
#from pypesto.sample.samplers.AdaptiveMetropolis.adaptive_metropolis_sampler import AdaptiveMetropolisSampler
from pypesto.sample.metropolis import AdaptiveMetropolisSampler
from pypesto.sample.samplers.parallel_tempering import parallel_tempering

from pypesto.visualize.sampling import *

import pypesto
import petab

import unittest
#import warnings

class SamplingTest(unittest.TestCase):
    def ks_options_AM(self):
        def logpdf(x):
            return norm.logpdf(x)

        sample = np.array([0])
        lower_bounds = np.array([-10])
        upper_bounds = np.array([10])
        covariance = np.identity(sample.size)

        # could change n_samples to chain_length to avoid confusion
        options = {
            'n_samples': 1000,
            #Adaptive Metropolis settings
            'debug': False,
            #'sample': sample,
            #'covariance': covariance,
            #'lower_bounds': lower_bounds,
            #'upper_bounds': upper_bounds,
            'decay_constant': 0.51,
            'threshold_sample': 1,
            'regularization_factor': 1e-6,
            #'log_posterior_callable': logpdf,
        }

        return options

    def ks_options_PT_AM(self, sampler, sampler_options):
        return sampler_options.update({
            #Parallel Tempering settings
            'sampler': sampler,
            'n_temperatures': 5,
            'exp_temperature': 4,
            'temperature_nu': 1000,
            'temperature_eta': 100,
            'max_temp': 50000
        })

    def test_adaptive_metropolis_sampler(self):
        sampler = AdaptiveMetropolisSampler(self.ks_options_AM())

        petab_problem = petab.Problem.from_yaml("doc/example/conversion_reaction/conversion_reaction.yaml")
        importer = pypesto.PetabImporter(petab_problem)
        problem = importer.create_problem()
        
        result = pypesto.minimize(problem, n_starts=10)
        result = pypesto.sample(problem, sampler, result=result)

        #ax = sampling_fval(result, problem, i_chain=0, burn_in=0, n_steps=1, figsize=None, fs = 12)
        ax = sampling_fval(result, problem, i_chain=0, stepsize=1, size=None, fs = 12)

        #result = sampler.sample()
        #test = kstest(result['samples'][0][0], 'norm')
        #self.assertTrue(test[0] < 0.2)

    #def test_parallel_tempering(self):
    #    result = parallel_tempering(self.ks_options_PT(AdaptiveMetropolisSampler, self.ks_options_AM()))
    #    test = kstest(result['samples'][0][0], 'norm')
    #    self.assertTrue(test[0] < 0.2)
