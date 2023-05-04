"""DynestySampler class."""

# TODO set `nlive` to > a few * n_params**2
#      - https://github.com/joshspeagle/dynesty/issues/301#issuecomment-888461184
#      - seems easy for `NestedSampler(..., live_points=...)`, and for
#        `DynamicNestedSampler(..., nlive0=...)`.
# TODO set one of the initial live points to the best vector from optimization
#      - this should hopefully ensure a good estimate for `L_max`, used for
#        stopping criteria
#      - seems easy for `NestedSampler(..., live_points)`, but not for
#        `DynamicNestedSampler`.
# TODO allow selection of bounding methods
#      - which bounding method works well for high-dimension problems?
#        - possibly overlapping balls/cubes (reduced tuning parameters), or
#          unit cube (but only if unimodal?)
#          - ellipsoids seem to be recommended
#        - requires benchmarking...
# TODO consider "inner" sampling method
#      - Hamiltonian for large problems (D > 20) with gradients
#        - need to reshape gradients to unit cube
#        - need to provide gradients, hopefully without additional obj call
#        - or use `compute_jac` option, probably expensive
#      - multivariate slice for large problems (D > 20) without gradients
# TODO consider reflective boundaries

from __future__ import annotations

import logging
from typing import List, Union

import numpy as np

from ..problem import Problem
from ..result import McmcPtResult
from .sampler import Sampler, SamplerImportError

logger = logging.getLogger(__name__)


class DynestySampler(Sampler):
    """Use dynesty for sampling.

    NB: `get_samples` returns MCMC-like samples, by resampling original
    `dynesty` samples according to their importance weights. This is because
    the original samples contain many low-likelihood samples.
    To work with the original samples, modify the results object with
    `pypesto_result.sample_result = sampler.get_original_samples()`, where
    `sampler` is an instance of `pypesto.sample.DynestySampler`. The original
    dynesty results object is available at `sampler.results`.

    NB: the dynesty samplers can be customized significantly, by providing
    `sampler_args` and `run_args` to your `pypesto.sample.DynestySampler()`
    call. For example, code to parallelize dynesty is provided in pyPESTO's
    `sampler_study.ipynb` notebook.

    Wrapper around https://dynesty.readthedocs.io/en/stable/, see there for
    details.
    """

    def __init__(
        self,
        sampler_args: dict = None,
        run_args: dict = None,
        dynamic: bool = True,
    ):
        """
        Initialize sampler.

        Parameters
        ----------
        sampler_args:
            Further keyword arguments that are passed on to the `__init__`
            method of the dynesty sampler.
        run_args:
            Further keyword arguments that are passed on to the `run_nested`
            method of the dynesty sampler.
        dynamic:
            Whether to use dynamic or static nested sampling.
        """
        # check dependencies
        import dynesty

        setup_dynesty()

        super().__init__()

        self.dynamic = dynamic

        if sampler_args is None:
            sampler_args = {}
        self.sampler_args: dict = sampler_args

        if run_args is None:
            run_args = {}
        self.run_args: dict = run_args

        # set in initialize
        self.problem: Union[Problem, None] = None
        self.sampler: Union[
            dynesty.DynamicNestedSampler,
            dynesty.NestedSampler,
            None,
        ] = None

    def prior_transform(self, prior_sample: np.ndarray) -> np.ndarray:
        """Transform prior sample from unit cube to pyPESTO prior.

        TODO support priors that are not uniform.
             raise warning in `self.initialize` for now.

        Parameters
        ----------
        prior_sample:
            The prior sample, provided by dynesty.
        problem:
            The pyPESTO problem.

        Returns
        -------
        The transformed prior sample.
        """
        return (
            prior_sample * (self.problem.ub - self.problem.lb)
            + self.problem.lb
        )

    def loglikelihood(self, x):
        """Log-probability density function."""
        # check if parameter lies within bounds
        if any(x < self.problem.lb) or any(x > self.problem.ub):
            return -np.inf
        # invert sign
        # TODO this is possibly the posterior if priors are defined
        return -1.0 * self.problem.objective(x)

    def initialize(
        self,
        problem: Problem,
        x0: Union[np.ndarray, List[np.ndarray]],
    ) -> None:
        """Initialize the sampler."""
        import dynesty

        setup_dynesty()

        self.problem = problem

        sampler_class = dynesty.NestedSampler
        if self.dynamic:
            sampler_class = dynesty.DynamicNestedSampler

        # initialize sampler
        self.sampler = sampler_class(
            loglikelihood=self.loglikelihood,
            prior_transform=self.prior_transform,
            ndim=len(self.problem.x_free_indices),
            **self.sampler_args,
        )

        # TODO somehow use optimized vector as one of the initial live points

    def sample(self, n_samples: int, beta: float = None) -> None:
        """Return the most recent sample state."""
        if n_samples is not None:
            logger.warning(
                "`n_samples` was specified but this is incompatible with "
                "`dynesty` samplers, as they run until other stopping "
                "criteria are satisfied."
            )
        if beta is not None:
            logger.warning(
                "The temperature of a `dynesty` sampler was set, but this is "
                "irrelevant for `dynesty` samplers."
            )

        self.sampler.run_nested(**self.run_args)
        self.results = self.sampler.results

    def save_internal_sampler(self, filename: str) -> None:
        """Save the state of the internal dynesty sampler.

        This makes it easier to analyze the original dynesty samples, after
        sampling, with `restore_internal`.

        Parameters
        ----------
        filename:
            The internal sampler will be saved here.
        """
        import dynesty

        setup_dynesty()

        dynesty.utils.save_sampler(
            sampler=self.sampler,
            fname=filename,
        )

    def restore_internal_sampler(self, filename: str) -> None:
        """Restore the state of the internal dynesty sampler.

        Parameters
        ----------
        filename:
            The internal sampler will be saved here.
        """
        import dynesty

        setup_dynesty()

        self.sampler = dynesty.utils.restore_sampler(fname=filename)

    def get_original_samples(self) -> McmcPtResult:
        """Get the samples into the fitting pypesto format.

        Returns
        -------
        The pyPESTO sample result.
        """
        return get_original_dynesty_samples(sampler=self.sampler)

    def get_samples(self) -> McmcPtResult:
        """Get MCMC-like samples into the fitting pypesto format.

        Returns
        -------
        The pyPESTO sample result.
        """
        return get_mcmc_like_dynesty_samples(sampler=self.sampler)


def get_original_dynesty_samples(sampler) -> McmcPtResult:
    """Get original dynesty samples.

    Parameters
    ----------
    sampler:
        The (internal!) dynesty sampler. See
        `pypesto.sample.DynestySampler.__init__`, specifically the
        `save_internal` argument, for more details.

    Returns
    -------
    The sample result.
    """
    trace_x = np.array([sampler.results.samples])
    trace_neglogpost = -np.array([sampler.results.logl])

    # the sampler uses custom adaptive priors
    trace_neglogprior = np.full(trace_neglogpost.shape, np.nan)
    # the sampler uses temperature 1
    betas = np.array([1.0])

    result = McmcPtResult(
        trace_x=trace_x,
        trace_neglogpost=trace_neglogpost,
        trace_neglogprior=trace_neglogprior,
        betas=betas,
    )

    return result


def get_mcmc_like_dynesty_samples(sampler) -> McmcPtResult:
    """Get MCMC-like samples.

    Parameters
    ----------
    sampler:
        The (internal!) dynesty sampler. See
        `pypesto.sample.DynestySampler.__init__`, specifically the
        `save_internal` argument, for more details.

    Returns
    -------
    The sample result.
    """
    import dynesty

    setup_dynesty()

    if len(sampler.results.importance_weights().shape) != 1:
        raise ValueError(
            "Unknown error. The dynesty importance weights are not a 1D array."
        )
    # resample according to importance weights
    indices = dynesty.utils.resample_equal(
        np.arange(sampler.results.importance_weights().shape[0]),
        sampler.results.importance_weights(),
    )

    trace_x = np.array([sampler.results.samples[indices]])
    trace_neglogpost = -np.array([sampler.results.logl[indices]])

    trace_neglogprior = np.array([np.full((len(indices),), np.nan)])
    betas = np.array([1.0])

    result = McmcPtResult(
        trace_x=trace_x,
        trace_neglogpost=trace_neglogpost,
        trace_neglogprior=trace_neglogprior,
        betas=betas,
    )
    return result


def setup_dynesty() -> None:
    """Import dynesty."""
    try:
        import dill  # noqa: S403
        import dynesty.utils

        dynesty.utils.pickle_module = dill
    except ImportError:
        raise SamplerImportError("dynesty")
