import numpy as np
from typing import Dict, Sequence, Union

from ..objective import Objective
from ..problem import Problem
from ..objective import History
from .sampler import InternalSample, InternalSampler
from .result import McmcPtResult


class MetropolisSampler(InternalSampler):
    """
    Simple Metropolis-Hastings sampler with fixed proposal variance.
    """

    def __init__(self, options: Dict = None):
        super().__init__(options)
        self.problem: Union[Problem, None] = None
        self.objective: Union[Objective, None] = None
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
        lpost = - self.trace_fval[-1]

        # loop over iterations
        for _ in range(int(n_samples)):
            # perform step
            x, lpost = self._perform_step(x, lpost, beta)

            # record step
            self.trace_x.append(x)
            self.trace_fval.append(-lpost)

    def _perform_step(self, x: np.ndarray, lpost: float, beta: float):
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
            prior_new = - np.inf
        else:
            # compute likelihood value
            llh_new = - self.objective(x_new)
            prior_new = - self.objective(x_new) # TODO

        # extract current likelihood value
        llh = lpost - prior # TODO
        # log acceptance probability (temper only likelihood)
        log_p_acc = min(beta * (llh_new - llh) + (prior_new - prior), 0)
        # flip a coin
        u = np.random.uniform(0, 1)
        # check acceptance
        if np.log(u) < log_p_acc:
            # update particle
            x = x_new
            # llh = llh_new
            lpost = llh_new + prior # TODO

        # update proposal
        self._update_proposal(x, log_p_acc, len(self.trace_fval)+1)

        return x, lpost

    def _propose_parameter(self, x: np.ndarray):
        """Propose a step."""
        x_new: np.ndarray = x + self.options['std'] * np.random.randn(len(x))
        return x_new

    def _update_proposal(self, x: np.ndarray, log_p_acc: float,
                         n_sample_cur: int):
        """Update the proposal density. Default: Do nothing."""

    def get_last_sample(self) -> InternalSample:
        return InternalSample(
            x=self.trace_x[-1],
            lpost=- self.trace_fval[-1]
        )

    def set_last_sample(self, sample: InternalSample):
        self.trace_x[-1] = sample.x
        self.trace_fval[-1] = - sample.lpost

    def get_samples(self) -> McmcPtResult:
        result = McmcPtResult(
            trace_x=np.array([self.trace_x]),
            trace_fval=np.array([self.trace_fval]),
            betas=np.array([1.]),
        )
        return result
