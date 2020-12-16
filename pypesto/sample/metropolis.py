import numpy as np
from typing import Dict, Sequence, Union
from tqdm import tqdm

from ..objective import History, ObjectiveBase, NegLogPriors
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
        self.neglogpost: Union[ObjectiveBase, None] = None
        self.neglogprior: Union[NegLogPriors, None] = None
        self.trace_x: Union[Sequence[np.ndarray], None] = None
        self.trace_neglogpost: Union[Sequence[float], None] = None
        self.trace_neglogprior: Union[Sequence[float], None] = None
        self.temper_lpost: bool = False

    @classmethod
    def default_options(cls):
        return {
            'std': 1.,  # the proposal standard deviation
            'show_progress': True,  # whether to show the progress
        }

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        self.neglogpost = problem.objective
        self.neglogpost.history = History()
        if problem.x_priors is None:
            self.neglogprior = lambda x: -0.
        else:
            self.neglogprior = problem.x_priors
        self.trace_x = [x0]
        self.trace_neglogpost = [self.neglogpost(x0)]
        self.trace_neglogprior = [self.neglogprior(x0)]

    def sample(self, n_samples: int, beta: float = 1.):
        # load last recorded particle
        x = self.trace_x[-1]
        lpost = - self.trace_neglogpost[-1]
        lprior = - self.trace_neglogprior[-1]

        show_progress = self.options['show_progress']

        # loop over iterations
        for _ in tqdm(range(int(n_samples)), disable=not show_progress):
            # perform step
            x, lpost, lprior = self._perform_step(
                x=x, lpost=lpost, lprior=lprior, beta=beta)
            # record step
            self.trace_x.append(x)
            self.trace_neglogpost.append(-lpost)
            self.trace_neglogprior.append(-lprior)

    def make_internal(self, temper_lpost: bool):
        self.options['show_progress'] = False
        self.temper_lpost = temper_lpost

    def _perform_step(self, x: np.ndarray, lpost: np.ndarray,
                      lprior: np.ndarray, beta: float):
        """
        Perform a step: Propose new parameter, evaluate and check whether to
        accept.
        """

        # propose step
        x_new: np.ndarray = self._propose_parameter(x)

        # check if step lies within bounds
        if any(x_new < self.problem.lb) or any(x_new > self.problem.ub):
            # will not be accepted
            lpost_new = - np.inf
        else:
            # compute log posterior
            lpost_new = - self.neglogpost(x_new)

        # check posterior evaluation is successful
        if np.isnan(lpost_new):
            # will not be accepted
            lpost_new = - np.inf

        # compute log prior
        lprior_new = - self.neglogprior(x_new)

        if not self.temper_lpost:
            # extract current log likelihood value
            llh = lpost - lprior
            # extract proposed log likelihood value
            llh_new = lpost_new - lprior_new
            # log acceptance probability (temper log likelihood)
            log_p_acc = min(beta * (llh_new - llh) + (lprior_new - lprior), 0)
        else:
            # log acceptance probability (temper log posterior)
            log_p_acc = min(beta * (lpost_new - lpost), 0)

        # flip a coin
        u = np.random.uniform(0, 1)

        # check acceptance
        if np.log(u) < log_p_acc:
            # update particle
            x = x_new
            lpost = lpost_new
            lprior = lprior_new

        # update proposal
        self._update_proposal(x, lpost,
                              log_p_acc, len(self.trace_neglogpost)+1)

        return x, lpost, lprior

    def _propose_parameter(self, x: np.ndarray):
        """Propose a step."""
        x_new: np.ndarray = x + self.options['std'] * np.random.randn(len(x))
        return x_new

    def _update_proposal(self, x: np.ndarray, lpost: float,
                         log_p_acc: float, n_sample_cur: int):
        """Update the proposal density. Default: Do nothing."""

    def get_last_sample(self) -> InternalSample:
        return InternalSample(
            x=self.trace_x[-1],
            lpost=- self.trace_neglogpost[-1],
            lprior=- self.trace_neglogprior[-1]
        )

    def set_last_sample(self, sample: InternalSample):
        self.trace_x[-1] = sample.x
        self.trace_neglogpost[-1] = - sample.lpost
        self.trace_neglogprior[-1] = - sample.lprior

    def get_samples(self) -> McmcPtResult:
        result = McmcPtResult(
            trace_x=np.array([self.trace_x]),
            trace_neglogpost=np.array([self.trace_neglogpost]),
            trace_neglogprior=np.array([self.trace_neglogprior]),
            betas=np.array([1.]),
        )
        return result
