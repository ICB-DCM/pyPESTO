"""Prior-based sampling."""

import logging

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
        priors: NegLogParameterPriors | NegLogPriors | None = None,
        max_iterations: int = 100,
    ) -> np.ndarray:
        """Sample startpoints from prior or uniform distribution.

        Parameters
        ----------
        n_starts: Number of startpoints to generate.
        lb: Lower parameter bound.
        ub: Upper parameter bound.
        priors: Parameter priors. If available, samples from priors;
            otherwise falls back to uniform sampling.
        max_iterations: Maximum number of resampling iterations to ensure
            all samples are within bounds.

        Returns
        -------
        xs: Startpoints, shape (n_starts, n_par).
        """
        # Fallback to uniform sampling if no priors available
        if priors is None:
            return uniform(n_starts=n_starts, lb=lb, ub=ub)

        n_par = len(lb)
        lb_reshaped = lb.reshape(1, -1)
        ub_reshaped = ub.reshape(1, -1)

        # Initialize samples and mark all rows for initial sampling
        samples = np.zeros((n_starts, n_par))
        rows_to_resample = np.ones(n_starts, dtype=bool)

        iteration = 0
        while True:
            n_resample = rows_to_resample.sum()
            if n_resample == 0:
                break

            # Generate new samples for the selected rows
            new_samples = np.zeros((n_resample, n_par))

            # Sample from priors where available
            prior_samples_dict = priors.sample(n_samples=n_resample)
            for param_index in prior_samples_dict.keys():
                new_samples[:, param_index] = prior_samples_dict[param_index]

            # Fill in uniform samples for parameters without priors
            for param_index in range(n_par):
                if param_index not in prior_samples_dict.keys():
                    new_samples[:, param_index] = np.random.uniform(
                        lb[param_index], ub[param_index], n_resample
                    )

            # Replace selected rows in the main array
            samples[rows_to_resample] = new_samples

            # Check bounds
            out_of_bounds = np.logical_or(
                samples < lb_reshaped, samples > ub_reshaped
            )

            # If all samples are within bounds, we are done
            if not np.any(out_of_bounds):
                break

            iteration += 1
            if iteration >= max_iterations:
                samples = np.clip(samples, lb_reshaped, ub_reshaped)
                logger.warning(
                    "Maximum startpoint resampling iterations reached. "
                    "Some samples were clipped to bounds."
                )
                break

            # Resample only rows that are still out-of-bounds
            rows_to_resample = np.any(out_of_bounds, axis=1)

        return samples
