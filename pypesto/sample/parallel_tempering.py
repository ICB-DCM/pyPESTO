import copy
from typing import Dict, List, Sequence, Union

import numpy as np
from tqdm import tqdm

from ..problem import Problem
from ..result import McmcPtResult
from .sampler import InternalSampler, Sampler


class ParallelTemperingSampler(Sampler):
    """Simple parallel tempering sampler.

    Parallel tempering is a Markov chain Monte Carlo (MCMC) method that
    uses multiple chains with different temperatures to sample from a
    probability distribution.
    The chains are coupled by swapping samples between them.
    This allows to sample from distributions with multiple modes more
    efficiently, as high-temperature chains can jump between modes, while
    low-temperature chains can sample the modes more precisely.

    This implementation is based on:

    * Vousden et al. 2016.
      Dynamic temperature selection for parallel tempering in Markov chain
      Monte Carlo simulations
      (https://doi.org/10.1093/mnras/stv2422),

    via a matlab-based reference implementation
    (https://github.com/ICB-DCM/PESTO/blob/master/private/performPT.m).
    """

    def __init__(
        self,
        internal_sampler: InternalSampler,
        betas: Sequence[float] = None,
        n_chains: int = None,
        options: Dict = None,
    ):
        super().__init__(options)

        # set betas
        if (betas is None) == (n_chains is None):
            raise ValueError("Set either betas or n_chains.")
        if betas is None:
            betas = near_exponential_decay_betas(
                n_chains=n_chains,
                exponent=self.options['exponent'],
                max_temp=self.options['max_temp'],
            )
        if betas[0] != 1.0:
            raise ValueError("The first chain must have beta=1.0")
        self.betas0 = np.array(betas)
        self.betas = None

        self.temper_lpost = self.options['temper_log_posterior']

        self.samplers = [
            copy.deepcopy(internal_sampler) for _ in range(len(self.betas0))
        ]
        # configure internal samplers
        for sampler in self.samplers:
            sampler.make_internal(temper_lpost=self.temper_lpost)

    @classmethod
    def default_options(cls) -> Dict:
        """Return the default options for the sampler."""
        return {
            'max_temp': 5e4,
            'exponent': 4,
            'temper_log_posterior': False,
            'show_progress': True,
        }

    def initialize(
        self, problem: Problem, x0: Union[np.ndarray, List[np.ndarray]]
    ):
        """Initialize all samplers."""
        n_chains = len(self.samplers)
        if isinstance(x0, list):
            x0s = x0
        else:
            x0s = [x0 for _ in range(n_chains)]
        for sampler, x0 in zip(self.samplers, x0s):
            _problem = copy.deepcopy(problem)
            sampler.initialize(_problem, x0)
        self.betas = self.betas0

    def sample(self, n_samples: int, beta: float = 1.0):
        """Sample and swap in between samplers."""
        show_progress = self.options['show_progress']
        # loop over iterations
        for i_sample in tqdm(range(int(n_samples)), disable=not show_progress):
            # TODO test
            # sample
            for sampler, beta in zip(self.samplers, self.betas):
                sampler.sample(n_samples=1, beta=beta)

            # swap samples
            swapped = self.swap_samples()

            # adjust temperatures
            self.adjust_betas(i_sample, swapped)

    def get_samples(self) -> McmcPtResult:
        """Concatenate all chains."""
        results = [sampler.get_samples() for sampler in self.samplers]
        trace_x = np.array([result.trace_x[0] for result in results])
        trace_neglogpost = np.array(
            [result.trace_neglogpost[0] for result in results]
        )
        trace_neglogprior = np.array(
            [result.trace_neglogprior[0] for result in results]
        )
        return McmcPtResult(
            trace_x=trace_x,
            trace_neglogpost=trace_neglogpost,
            trace_neglogprior=trace_neglogprior,
            betas=self.betas,
        )

    def swap_samples(self) -> Sequence[bool]:
        """Swap samples as in Vousden2016."""
        # for recording swaps
        swapped = []

        if len(self.betas) == 1:
            # nothing to be done
            return swapped

        # beta differences
        dbetas = self.betas[:-1] - self.betas[1:]

        # loop over chains from highest temperature down
        for dbeta, sampler1, sampler2 in reversed(
            list(zip(dbetas, self.samplers[:-1], self.samplers[1:]))
        ):
            # extract samples
            sample1 = sampler1.get_last_sample()
            sample2 = sampler2.get_last_sample()

            # extract log likelihood values
            sample1_llh = sample1.lpost - sample1.lprior
            sample2_llh = sample2.lpost - sample2.lprior

            # swapping probability
            p_acc_swap = dbeta * (sample2_llh - sample1_llh)

            # flip a coin
            u = np.random.uniform(0, 1)

            # check acceptance
            swap = np.log(u) < p_acc_swap
            if swap:
                # swap
                sampler2.set_last_sample(sample1)
                sampler1.set_last_sample(sample2)

            # record
            swapped.insert(0, swap)
        return swapped

    def adjust_betas(self, i_sample: int, swapped: Sequence[bool]):
        """Adjust temperature values. Default: Do nothing."""


def near_exponential_decay_betas(
    n_chains: int, exponent: float, max_temp: float
) -> np.ndarray:
    """Initialize betas in a near-exponential decay scheme.

    Parameters
    ----------
    n_chains:
        Number of chains to use.
    exponent:
        Decay exponent. The higher, the more small temperatures are used.
    max_temp:
        Maximum chain temperature.
    """
    # special case of one chain
    if n_chains == 1:
        return np.array([1.0])

    temperatures = (
        np.linspace(1, max_temp ** (1 / exponent), n_chains) ** exponent
    )
    betas = 1 / temperatures

    return betas
