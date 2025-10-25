"""Prior-based sampling."""

import logging
from typing import Union

import numpy as np

from ..objective import NegLogParameterPriors, NegLogPriors
from .base import CheckedStartpoints
from .uniform import uniform

logger = logging.getLogger(__name__)


class PriorStartpoints(CheckedStartpoints):
    """Generate startpoints from prior distribution.

    Samples from the prior distribution if available, otherwise falls back
    to uniform sampling. Ensures all samples are within bounds by resampling
    out-of-bounds values.
    """

    def __init__(
        self,
        use_guesses: bool = True,
        check_fval: bool = False,
        check_grad: bool = False,
    ):
        """Initialize.

        Parameters
        ----------
        use_guesses:
            Whether to use guesses provided in the problem.
        check_fval:
            Whether to check function values at the startpoint, and resample
            if not finite.
        check_grad:
            Whether to check gradients at the startpoint, and resample
            if not finite.
        """
        super().__init__(
            use_guesses=use_guesses,
            check_fval=check_fval,
            check_grad=check_grad,
        )

    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
        priors: Union[NegLogParameterPriors, NegLogPriors, None] = None,
    ) -> np.ndarray:
        """Sample startpoints from prior or uniform distribution.

        Parameters
        ----------
        n_starts: Number of startpoints to generate.
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        priors: Parameter priors. If available, samples from priors;
            otherwise falls back to uniform sampling. For parameters
            without priors, uniform sampling is used.

        Returns
        -------
        xs: Startpoints, shape (n_starts, n_par).
        """
        # If priors are available, use prior.sample()
        if priors is not None:
            # Initialize full array with correct number of parameters
            n_par = len(lb)
            samples = np.zeros((n_starts, n_par))

            # Identify which parameters have priors
            prior_indices = {prior["index"] for prior in priors.prior_list}

            # Sample from priors where available
            prior_samples = priors.sample(n_samples=n_starts)
            for i in prior_indices:
                samples[:, i] = prior_samples[:, i]

            # Fill in uniform samples for parameters without priors
            for i in range(n_par):
                if i not in prior_indices:
                    samples[:, i] = np.random.uniform(lb[i], ub[i], n_starts)

            # Check bounds and resample out-of-bounds values
            samples = self._resample_out_of_bounds(
                samples, lb, ub, priors, prior_indices
            )

            return samples

        # Fallback to uniform sampling if no priors available
        return uniform(n_starts=n_starts, lb=lb, ub=ub)

    def _resample_out_of_bounds(
        self,
        samples: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        priors: Union[NegLogParameterPriors, NegLogPriors],
        prior_indices: set,
    ) -> np.ndarray:
        """Resample any samples that are out of bounds.

        Parameters
        ----------
        samples: Current samples, shape (n_samples, n_par).
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        priors: Parameter priors for resampling.
        prior_indices: Set of parameter indices that have priors.

        Returns
        -------
        samples: Samples with out-of-bounds values replaced, shape (n_samples, n_par).
        """
        lb_reshaped = lb.reshape(1, -1)
        ub_reshaped = ub.reshape(1, -1)

        # Identify out-of-bounds samples
        out_of_bounds = np.logical_or(
            samples < lb_reshaped, samples > ub_reshaped
        )

        # Resample until all samples are within bounds
        max_iterations = 1000
        iteration = 0

        while np.any(out_of_bounds) and iteration < max_iterations:
            # Identify which samples need resampling
            rows_to_resample = np.any(out_of_bounds, axis=1)
            n_resample = rows_to_resample.sum()

            if n_resample == 0:
                break

            # Initialize full array for new samples
            n_par = len(lb)
            new_samples = np.zeros((n_resample, n_par))

            # Generate new samples from priors
            prior_samples = priors.sample(n_samples=n_resample)
            for i in prior_indices:
                new_samples[:, i] = prior_samples[:, i]

            # Fill in uniform samples for parameters without priors
            for i in range(n_par):
                if i not in prior_indices:
                    new_samples[:, i] = np.random.uniform(
                        lb[i], ub[i], n_resample
                    )

            # Replace out-of-bounds samples
            samples[rows_to_resample] = new_samples

            # Check bounds again
            out_of_bounds = np.logical_or(
                samples < lb_reshaped, samples > ub_reshaped
            )
            iteration += 1

        if iteration >= max_iterations:
            # If we still have out-of-bounds samples after max iterations,
            # clip them to bounds
            samples = np.clip(samples, lb_reshaped, ub_reshaped)
            logger.warning(
                "Maximum startpoint resampling iterations reached. "
                "Some samples were clipped to bounds."
            )

        return samples
