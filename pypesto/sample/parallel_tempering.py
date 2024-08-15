import copy
import logging
from collections.abc import Sequence
from typing import Union

import numpy as np

from ..C import BETA_DECAY, EXPONENTIAL_DECAY
from ..problem import Problem
from ..result import McmcPtResult, Result
from ..util import tqdm
from .diagnostics import geweke_test
from .sampler import InternalSampler, Sampler

logger = logging.getLogger(__name__)


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
        options: dict = None,
    ):
        super().__init__(options)

        # set betas
        if (betas is None) == (n_chains is None):
            raise ValueError("Set either betas or n_chains.")
        if betas is None and self.options["beta_init"] == EXPONENTIAL_DECAY:
            logger.info('Initializing betas with "near-exponential decay".')
            betas = near_exponential_decay_betas(
                n_chains=n_chains,
                exponent=self.options["exponent"],
                max_temp=self.options["max_temp"],
            )
        elif betas is None and self.options["beta_init"] == BETA_DECAY:
            logger.info('Initializing betas with "beta decay".')
            betas = beta_decay_betas(
                n_chains=n_chains, alpha=self.options["alpha"]
            )
        if betas[0] != 1.0:
            raise ValueError("The first chain must have beta=1.0")
        self.betas0 = np.array(betas)
        self.betas = None

        self.temper_lpost = self.options["temper_log_posterior"]

        self.samplers = [
            copy.deepcopy(internal_sampler) for _ in range(len(self.betas0))
        ]
        # configure internal samplers
        for sampler in self.samplers:
            sampler.make_internal(temper_lpost=self.temper_lpost)

    @classmethod
    def default_options(cls) -> dict:
        """Return the default options for the sampler."""
        return {
            "max_temp": 5e4,
            "exponent": 4,
            "temper_log_posterior": False,
            "show_progress": None,
            "beta_init": BETA_DECAY,  # replaced in adaptive PT
            "alpha": 0.3,
        }

    def initialize(
        self, problem: Problem, x0: Union[np.ndarray, list[np.ndarray]]
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
        show_progress = self.options.get("show_progress", None)
        # loop over iterations
        for i_sample in tqdm(range(int(n_samples)), enable=show_progress):
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

    def compute_log_evidence(
        self,
        result: Result,
        method: str = "trapezoid",
        use_all_chains: bool = True,
    ) -> Union[float, None]:
        """Perform thermodynamic integration to estimate the log evidence.

        Parameters
        ----------
        result:
            Result object containing the samples.
        method:
            Integration method, either 'trapezoid' or 'simpson' (uses scipy for integration).
        use_all_chains:
            If True, calculate burn-in for each chain and use the maximal burn-in for all chains for the integration.
            This will fail if not all chains have converged yet.
            Otherwise, use only the converged chains for the integration (might increase the integration error).
        """
        from scipy.integrate import simpson, trapezoid

        if self.options["beta_init"] == EXPONENTIAL_DECAY:
            logger.warning(
                "The temperature schedule is not optimal for thermodynamic integration. "
                f"Carefully check the results. Consider using beta_init='{BETA_DECAY}' for better results."
            )

        # compute burn in for all chains but the last one (prior only)
        burn_ins = np.zeros(len(self.betas), dtype=int)
        for i_chain in range(len(self.betas)):
            burn_ins[i_chain] = geweke_test(result, chain_number=i_chain)
        max_burn_in = int(np.max(burn_ins))

        if max_burn_in >= result.sample_result.trace_x.shape[1]:
            logger.warning(
                f"At least {np.sum(burn_ins >= result.sample_result.trace_x.shape[1])} chains seem to not have "
                f"converged yet. You may want to use a larger number of samples."
            )
            if use_all_chains:
                raise ValueError(
                    "Not all chains have converged yet. You may want to use a larger number of samples, "
                    "or try ´use_all_chains=False´, which might increase the integration error."
                )

        if use_all_chains:
            # estimate mean of log likelihood for each beta
            trace_loglike = (
                result.sample_result.trace_neglogprior[::-1, max_burn_in:]
                - result.sample_result.trace_neglogpost[::-1, max_burn_in:]
            )
            mean_loglike_per_beta = np.mean(trace_loglike, axis=1)
            temps = self.betas[::-1]
        else:
            # estimate mean of log likelihood for each beta if chain has converged
            mean_loglike_per_beta = []
            temps = []
            for i_chain in reversed(range(len(self.betas))):
                if burn_ins[i_chain] < result.sample_result.trace_x.shape[1]:
                    # save temperature-chain as it is converged
                    temps.append(self.betas[i_chain])
                    # calculate mean log likelihood for each beta
                    trace_loglike_i = (
                        result.sample_result.trace_neglogprior[
                            i_chain, burn_ins[i_chain] :
                        ]
                        - result.sample_result.trace_neglogpost[
                            i_chain, burn_ins[i_chain] :
                        ]
                    )
                    mean_loglike_per_beta.append(np.mean(trace_loglike_i))

        if method == "trapezoid":
            log_evidence = trapezoid(
                # integrate from low to high temperature
                y=mean_loglike_per_beta,
                x=temps,
            )
        elif method == "simpson":
            log_evidence = simpson(
                # integrate from low to high temperature
                y=mean_loglike_per_beta,
                x=temps,
            )
        else:
            raise ValueError(
                f"Unknown method {method}. Choose 'trapezoid' or 'simpson'."
            )

        return log_evidence


def beta_decay_betas(n_chains: int, alpha: float) -> np.ndarray:
    """Initialize betas to the (j-1)th quantile of a Beta(alpha, 1) distribution.

    Proposed by Xie et al. (2011) to be used for thermodynamic integration.

    Parameters
    ----------
    n_chains:
        Number of chains to use.
    alpha:
        Tuning parameter that modulates the skew of the distribution over the temperatures.
        For alpha=1 we have a uniform distribution, and as alpha decreases towards zero,
        temperatures become positively skewed. Xie et al. (2011) propose alpha=0.3 as a good start.
    """
    if alpha <= 0 or alpha > 1:
        raise ValueError("alpha must be in (0, 1]")

    # special case of one chain
    if n_chains == 1:
        return np.array([1.0])

    return np.power(np.arange(n_chains) / (n_chains - 1), 1 / alpha)[::-1]


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
