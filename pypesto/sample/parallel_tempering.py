from typing import Any, Dict, Sequence, Union
import numpy as np
import copy

from ..problem import Problem
from .sampler import Sampler, TemperableSampler
from .result import McmcPtResult


class ParallelTemperingSampler(Sampler):

    def __init__(
            self,
            internal_sampler: TemperableSampler,
            options: Dict = None):
        self.internal_sampler = internal_sampler
        self.options = ParallelTemperingSampler.translate_options(options)
        self.samplers = [copy.deepcopy(self.internal_sampler)
                         for _ in range(self.options['n_chains'])]
        self.betas = np.array(self.options['betas'])

    @classmethod
    def translate_options(cls, options):
        used_options = {
            'n_chains': 3,
            'betas': [1/1e0, 1/1e1, 1/1e3],
        }
        if options is None:
            options = {}
        for key, val in options:
            if key not in used_options:
                raise KeyError(f"Cannot handle key {key}.")
            used_options[key] = val
        if used_options['n_chains'] != len(used_options['betas']):
            raise AssertionError(
                "Numbers of chains and temperatures do not match.")
        return used_options

    def initialize(self, problem: Problem, x0: np.ndarray):
        for sampler in self.samplers:
            _problem = copy.deepcopy(problem)
            sampler.initialize(_problem, x0)

    def sample(
            self, n_samples: int, beta: float = 1.):
        # loop over iterations
        for _ in range(int(n_samples)):
            # sample
            for sampler, beta in zip(self.samplers, self.betas):
                sampler.sample(n_samples=1, beta=beta)

            # swap samples
            ParallelTemperingSampler.swap_samples(self.samplers, self.betas)

    def get_samples(self) -> McmcPtResult:
        results = [sampler.get_samples() for sampler in self.samplers]
        trace_x = np.array([result.trace_x[0] for result in results])
        trace_fval = np.array([result.trace_fval[0] for result in results])
        return McmcPtResult(
            trace_x=trace_x,
            trace_fval=trace_fval,
            betas=self.betas
        )

    @staticmethod
    def swap_samples(
            samplers: Sequence[TemperableSampler],
            betas: np.ndarray):
        """Swap samples."""
        n_chains = len(betas)
        if n_chains == 1:
            return
        dbetas = betas[:-1] - betas[1:]
        for dbeta, sampler1, sampler2 in reversed(
                list(zip(dbetas, samplers[:-1], samplers[1:]))):
            sample1 = sampler1.get_last_sample()
            sample2 = sampler2.get_last_sample()
            p_acc_swap = dbeta * ( sample2.llh - sample1.llh)

            u = np.random.uniform(0, 1)
            if np.log(u) < p_acc_swap:
                sampler2.set_last_sample(sample1)
                sampler1.set_last_sample(sample2)
                # TODO implement loggers