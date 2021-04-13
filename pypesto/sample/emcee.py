from typing import List, Union
import numpy as np

from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult
from ..startpoint import assign_startpoints, uniform

try:
    import emcee
except ImportError:
    emcee = None


class EmceeSampler(Sampler):
    """Use emcee for sampling.

    Wrapper around https://emcee.readthedocs.io/en/stable/, see there for
    details.
    """

    def __init__(
        self,
        nwalkers: int = 1,
        sampler_args: dict = None,
        run_args: dict = None,
    ):
        """
        Parameters
        ----------
        nwalkers: The number of walkers in the ensemble.
        sampler_args:
            Further keyword arguments that are passed on to
            ``emcee.EnsembleSampler.__init__``.
        run_args:
            Further keyword arguments that are passed on to
            ``emcee.EnsembleSampler.run_mcmc``.
        """
        # check dependencies
        if emcee is None:
            raise ImportError(
                "This sampler requires an installation of emcee. Install e.g. "
                "via ``pip install pypesto[emcee]``.")

        super().__init__()
        self.nwalkers: int = nwalkers

        if sampler_args is None:
            sampler_args = {}
        self.sampler_args: dict = sampler_args

        if run_args is None:
            run_args = {}
        self.run_args: dict = run_args

        # set in initialize
        self.problem: Union[Problem, None] = None
        self.sampler: Union[emcee.EnsembleSampler, None] = None
        self.state: Union[emcee.State, None] = None

    def initialize(
        self,
        problem: Problem,
        x0: Union[np.ndarray, List[np.ndarray]],
    ) -> None:
        self.problem = problem

        # extract for pickling efficiency
        objective = self.problem.objective
        lb = self.problem.lb
        ub = self.problem.ub

        # parameter dimenstion
        ndim = len(self.problem.x_free_indices)

        def log_prob(x):
            """Log-probability density function."""
            # check if parameter lies within bounds
            if any(x < lb) or any(x > ub):
                return - np.inf
            # invert sign
            return - 1. * objective(x)

        # initialize sampler
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=ndim,
            log_prob_fn=log_prob,
            **self.sampler_args,
        )

        # assign startpoints
        if self.state is None:
            #  extract x0
            x0 = np.asarray(x0)
            if x0.ndim == 1:
                x0 = [x0]
            x0 = np.array([problem.get_full_vector(x) for x in x0])
            #  add x0 to guesses
            problem.x_guesses_full = np.row_stack((x0, problem.x_guesses_full))

            #  sample start points
            self.state = assign_startpoints(
                n_starts=self.nwalkers, startpoint_method=uniform,
                problem=problem, startpoint_resample=True)

            #  restore original guesses
            problem.x_guesses_full = problem.x_guesses_full[x0.shape[0]:]

    def sample(self, n_samples: int, beta: float = 1.) -> None:
        # the method returns the most recent sample state
        self.state = self.sampler.run_mcmc(
            self.state, n_samples, **self.run_args)

    def get_samples(self) -> McmcPtResult:
        # all walkers are concatenated, yielding a flat array
        trace_x = np.array([self.sampler.get_chain(flat=True)])
        trace_neglogpost = np.array([- self.sampler.get_log_prob(flat=True)])
        # the sampler does not know priors
        trace_neglogprior = np.full(trace_neglogpost.shape, np.nan)
        # the walkers all run on temperature 1
        betas = np.array([1.])

        result = McmcPtResult(
            trace_x=trace_x,
            trace_neglogpost=trace_neglogpost,
            trace_neglogprior=trace_neglogprior,
            betas=betas,
        )

        return result
