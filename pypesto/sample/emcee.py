"""EmceeSampler class."""

from __future__ import annotations

import logging

import numpy as np

from ..problem import Problem
from ..result import McmcPtResult
from ..startpoint import UniformStartpoints, uniform
from .sampler import Sampler, SamplerImportError

logger = logging.getLogger(__name__)


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
        Initialize sampler.

        Parameters
        ----------
        nwalkers:
            The number of walkers in the ensemble.
        sampler_args:
            Further keyword arguments that are passed on to
            ``emcee.EnsembleSampler.__init__``.
        run_args:
            Further keyword arguments that are passed on to
            ``emcee.EnsembleSampler.run_mcmc``.
        """
        # check dependencies
        try:
            import emcee
        except ImportError:
            raise SamplerImportError("emcee") from None

        super().__init__()
        self.nwalkers: int = nwalkers

        if sampler_args is None:
            sampler_args = {}
        self.sampler_args: dict = sampler_args

        if run_args is None:
            run_args = {}
        self.run_args: dict = run_args

        # set in initialize
        self.problem: Problem | None = None
        self.sampler: emcee.EnsembleSampler | None = None
        self.state: emcee.State | None = None

    def get_epsilon_ball_initial_state(
        self,
        center: np.ndarray,
        problem: Problem,
        epsilon: float = 1e-3,
    ):
        """Get walker initial positions as samples from an epsilon ball.

        The ball is scaled in each direction according to the magnitude of the
        center in that direction.

        It is assumed that, because vectors are generated near a good point,
        all generated vectors are evaluable, so evaluability is not checked.

        Points that are generated outside the problem bounds will get shifted
        to lie on the edge of the problem bounds.

        Parameters
        ----------
        center:
            The center of the epsilon ball. The dimension should match the full
            dimension of the pyPESTO problem. This will be returned as the
            first position.
        problem:
            The pyPESTO problem.
        epsilon:
            The relative radius of the ball. e.g., if `epsilon=0.5`
            and the center of the first dimension is at 100, then the upper
            and lower bounds of the epsilon ball in the first dimension will
            be 150 and 50, respectively.
        """
        # Epsilon ball
        lb = center * (1 - epsilon)
        ub = center * (1 + epsilon)

        # Adjust bounds to satisfy problem bounds
        lb[lb < problem.lb] = problem.lb[lb < problem.lb]
        ub[ub > problem.ub] = problem.ub[ub > problem.ub]

        # Sample initial positions
        initial_state_after_first = uniform(
            n_starts=self.nwalkers - 1,
            lb=lb,
            ub=ub,
        )

        # Include `center` in initial positions
        initial_state = np.row_stack(
            (
                center,
                initial_state_after_first,
            )
        )

        return initial_state

    def initialize(
        self,
        problem: Problem,
        x0: np.ndarray | list[np.ndarray],
    ) -> None:
        """Initialize the sampler.

        It is recommended to initialize walkers

        Parameters
        ----------
        x0:
            The "a priori preferred position". e.g., an optimized parameter
            vector. https://emcee.readthedocs.io/en/stable/user/faq/
            The position of the first walker will be this, the remaining
            walkers will be assigned positions uniformly in a smaller ball
            around this vector.
            Alternatively, a set of vectors can be provided, which will be used
            to initialize walkers. In this case, any remaining walkers will be
            initialized at points sampled uniformly within the problem bounds.
        """
        import emcee

        self.problem = problem

        # extract for pickling efficiency
        objective = self.problem.objective
        lb = self.problem.lb
        ub = self.problem.ub

        # parameter dimension
        ndim = len(self.problem.x_free_indices)

        def log_prob(x):
            """Log-probability density function."""
            # check if parameter lies within bounds
            if any(x < lb) or any(x > ub):
                return -np.inf
            # invert sign
            return -1.0 * objective(x)

        # initialize sampler
        self.sampler = emcee.EnsembleSampler(
            nwalkers=self.nwalkers,
            ndim=ndim,
            log_prob_fn=log_prob,
            **self.sampler_args,
        )

        # assign startpoints
        if self.state is None:
            if x0.ndim > 1 and len(x0.shape[0]) > 1:
                logger.warning(
                    "More than a single vector was provided to initialize the "
                    "walker positions. If these vectors do not exist in a "
                    "small ball around a high-probability position (e.g. "
                    "optimized vector) then sampling may be inefficient (see "
                    "emcee FAQ: "
                    "https://emcee.readthedocs.io/en/stable/user/faq/ )."
                )
                #  extract x0
                x0 = np.asarray(x0)
                if x0.ndim == 1:
                    x0 = [x0]
                x0 = np.array([problem.get_full_vector(x) for x in x0])
                x_guesses_full0 = problem.x_guesses_full
                #  add x0 to guesses
                problem.set_x_guesses(
                    np.row_stack(
                        (
                            x0,
                            problem.x_guesses_full,
                        )
                    )
                )
                #  sample start points
                initial_state = UniformStartpoints(
                    use_guesses=True,
                    check_fval=True,
                    check_grad=False,
                )(
                    n_starts=self.nwalkers,
                    problem=problem,
                )
                #  restore original guesses
                problem.set_x_guesses(x_guesses_full0)
            else:
                initial_state = self.get_epsilon_ball_initial_state(
                    center=x0,
                    problem=problem,
                )

            self.state = initial_state

    def sample(self, n_samples: int, beta: float = 1.0) -> None:
        """Return the most recent sample state."""
        self.state = self.sampler.run_mcmc(
            initial_state=self.state,
            nsteps=n_samples,
            **self.run_args,
        )

    def get_samples(self) -> McmcPtResult:
        """Get the samples into the fitting pypesto format."""
        # all walkers are concatenated, yielding a flat array
        trace_x = np.array([self.sampler.get_chain(flat=True)])
        trace_neglogpost = np.array([-self.sampler.get_log_prob(flat=True)])
        # the sampler does not know priors
        trace_neglogprior = np.full(trace_neglogpost.shape, np.nan)
        # the walkers all run on temperature 1
        betas = np.array([1.0])

        result = McmcPtResult(
            trace_x=trace_x,
            trace_neglogpost=trace_neglogpost,
            trace_neglogprior=trace_neglogprior,
            betas=betas,
        )

        return result
