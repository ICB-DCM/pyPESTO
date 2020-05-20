import numpy as np
from typing import Dict, Sequence, Union

from ..objective import History
from ..objective.objective import ObjectiveBase
from ..problem import Problem
from .sampler import InternalSample, InternalSampler
from .result import McmcPtResult


class MetropolisSampler(InternalSampler):
    """
    Simple Metropolis-Hastings sampler with fixed proposal variance.
    """

    def __init__(self, options: Dict = None):
        super().__init__(options)
        self.problem: Union[Problem, None] = None
        self.objective: Union[ObjectiveBase, None] = None
        self.trace_x: Union[Sequence[np.ndarray], None] = None
        self.trace_fval: Union[Sequence[float], None] = None

    @classmethod
    def default_options(cls):
        return {
            'std': 1.,  # the proposal standard deviation
        }

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        self.objective = problem.objective
        self.objective.history = History()
        self.trace_x = [x0]
        self.trace_fval = [self.objective(x0)]

    def sample(self, n_samples: int, beta: float = 1.):
        # load last recorded particle
        x = self.trace_x[-1]
        llh = - self.trace_fval[-1]

        # loop over iterations
        for _ in range(int(n_samples)):
            # perform step
            x, llh = self._perform_step(x, llh, beta)

            # record step
            self.trace_x.append(x)
            self.trace_fval.append(-llh)

    def _perform_step(self, x: np.ndarray, llh: float, beta: float):
        """
        Perform a step: Propose new parameter, evaluate and check whether to
        accept.
        """
        # propose step
        x_new: np.ndarray = self._propose_parameter(x)

        # check if step lies within bounds
        if any(x_new < self.problem.lb) or any(x_new > self.problem.ub):
            # will not be accepted
            llh_new = - np.inf
        else:
            # compute function value
            llh_new = - self.objective(x_new)

        # log acceptance probability
        log_p_acc = min(beta * (llh_new - llh), 0)

        # flip a coin
        u = np.random.uniform(0, 1)
        # check acceptance
        if np.log(u) < log_p_acc:
            # update particle
            x = x_new
            llh = llh_new

        # update proposal
        self._update_proposal(x, llh, log_p_acc, len(self.trace_fval)+1)

        return x, llh

    def _propose_parameter(self, x: np.ndarray):
        """Propose a step."""
        x_new: np.ndarray = x + self.options['std'] * np.random.randn(len(x))
        return x_new

    def _update_proposal(self, x: np.ndarray, llh: float, log_p_acc: float,
                         n_sample_cur: int):
        """Update the proposal density. Default: Do nothing."""

    def get_last_sample(self) -> InternalSample:
        return InternalSample(
            x=self.trace_x[-1],
            llh=- self.trace_fval[-1]
        )

    def set_last_sample(self, sample: InternalSample):
        self.trace_x[-1] = sample.x
        self.trace_fval[-1] = - sample.llh

    def get_samples(self) -> McmcPtResult:
        result = McmcPtResult(
            trace_x=np.array([self.trace_x]),
            trace_fval=np.array([self.trace_fval]),
            betas=np.array([1.]),
        )
        return result
