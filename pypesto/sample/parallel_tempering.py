from typing import Any, Dict, Union
import numpy as np

from ..problem import Problem
from .sampler import Sampler, InternalSampler
from .result import McmcPtResult


class ParallelTemperingSampler(Sampler):

    def __init__(
            self,
            internal_sampler: InternalSampler,
            options: Dict = None):
        self.internal_sampler = internal_sampler
        self.options = ParallelTemperingSampler.translate_options(options)

    @staticmethod
    def translate_options(options):
        default_options = {
            'n_samples': 1000,
            'n_chains': 10,
        }
        if options is None:
            options = {}
        for key, val in options:
            if key not in default_options:
                raise KeyError(f"Cannot handle key {key}.")
            default_options[key] = val
        return default_options

    def sample(
            self, problem: Problem, x0: np.ndarray = None
    ) -> Union[McmcPtResult, Any]:
        n_chains = self.options['n_chains']

        # initialize temperatures
        # internal_samplers = [copy.deepcopy(self.internal_sampler)
        #                      for _ in range(n_chains)]

        # loop over iterations
        #     loop over temperatures
        #         internal_samplers[i_chain].perform_step(...)

