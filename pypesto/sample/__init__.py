"""
Sample
======

"""

from .sample import sample
from .sampler import Sampler, TemperableSampler
from .pymc3 import Pymc3Sampler
from .metropolis import MetropolisSampler
from .parallel_tempering import ParallelTemperingSampler
from .result import McmcPtResult