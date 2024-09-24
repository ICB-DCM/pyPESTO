# noqa: D400,D205
"""
Sample
======

Draw samples from the distribution, with support for various samplers.
"""

from .adaptive_metropolis import AdaptiveMetropolisSampler
from .adaptive_parallel_tempering import AdaptiveParallelTemperingSampler
from .diagnostics import auto_correlation, effective_sample_size, geweke_test
from .dynesty import DynestySampler
from .emcee import EmceeSampler
from .evidence import (
    bridge_sampling_log_evidence,
    harmonic_mean_log_evidence,
    laplace_approximation_log_evidence,
    parallel_tempering_log_evidence,
)
from .metropolis import MetropolisSampler
from .parallel_tempering import ParallelTemperingSampler
from .sample import sample
from .sampler import InternalSampler, Sampler
from .util import calculate_ci_mcmc_sample, calculate_ci_mcmc_sample_prediction
