"""AdaptiveParallelTemperingSampler class."""

from collections.abc import Sequence

import numpy as np

from ..C import EXPONENTIAL_DECAY
from .parallel_tempering import ParallelTemperingSampler


class AdaptiveParallelTemperingSampler(ParallelTemperingSampler):
    """Parallel tempering sampler with adaptive temperature adaptation.

    Compared to the base class, this sampler adapts the temperatures
    during the sampling process.
    This both simplifies the setup as it avoids manual tuning,
    and improves the performance as the temperatures are adapted to the
    current state of the chains.

    This implementation is based on:

    * Vousden et al. 2016.
      Dynamic temperature selection for parallel tempering in Markov chain
      Monte Carlo simulations
      (https://doi.org/10.1093/mnras/stv2422),

    via a matlab reference implementation
    (https://github.com/ICB-DCM/PESTO/blob/master/private/performPT.m).
    """

    @classmethod
    def default_options(cls) -> dict:
        """Get default options for sampler."""
        options = super().default_options()
        # scaling factor for temperature adaptation
        options["eta"] = 100
        # controls the adaptation degeneration velocity of the temperature
        # adaption.
        options["nu"] = 1e3
        # initial temperature schedule as in Vousden et. al. 2016.
        options["beta_init"] = EXPONENTIAL_DECAY

        return options

    def adjust_betas(self, i_sample: int, swapped: Sequence[bool]):
        """Update temperatures as in Vousden2016."""
        if len(self.betas) == 1:
            return

        # parameters
        nu = self.options["nu"]
        eta = self.options["eta"]
        betas = self.betas

        # booleans to integer array
        swapped = np.array([int(swap) for swap in swapped])

        # update betas
        kappa = nu / (i_sample + 1 + nu) / eta
        ds = kappa * (swapped[:-1] - swapped[1:])
        dtemp = np.diff(1.0 / betas[:-1])
        dtemp = dtemp * np.exp(ds)
        betas[:-1] = 1 / np.cumsum(np.insert(dtemp, obj=0, values=1.0))

        # fill in
        self.betas = betas
