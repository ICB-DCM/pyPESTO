import numpy as np
from typing import Any, Callable, Dict, Union, Tuple
import time

from ..objective import Objective
from ..problem import Problem
from ..objective import History
from .sampler import Sampler, InternalSampler
from .result import McmcPtResult


class MetropolisSampler(Sampler, InternalSampler):
    """Simple Metropolis-Hastings sampler with fixed proposal variance."""

    def __init__(self, options: Dict = None):
        self.options = MetropolisSampler.translate_options(options)

    @staticmethod
    def translate_options(options):
        default_options = {
            'n_samples': 1000
        }
        if options is None:
            options = {}
        for key, val in options:
            if key not in default_options:
                raise KeyError(f"Cannot handle key {key}.")
            default_options[key] = val
        return default_options

    def sample(
            self, problem: Problem, x0: np.ndarray = None
    ) -> Union[McmcPtResult, Any]:
        trace_x = []
        trace_fval = []

        # set up objective history
        objective = problem.objective
        objective.history = History()

        start_time = time.time()

        x = x0
        llh = - objective(x)

        for _ in range(self.options['n_samples']):
            x, llh = self.perform_step(x, llh, objective)

            trace_x.append(x)
            trace_fval.append(-llh)

        result = McmcPtResult(
            trace_x=[trace_x],
            trace_fval=[trace_fval],
            trace_grad=[[None]*len(trace_fval)],
            temperatures=[1],
            time=time.time()-start_time,
            n_fval=objective.history.n_fval,
            n_grad=objective.history.n_grad,
            n_hess=objective.history.n_hess
        )

        return result

    def perform_step(self, x: np.ndarray, llh: float, objective: Objective):
        x_new = x + 0.5 * np.random.randn(len(x))
        llh_new = - objective(x_new)

        u = np.random.uniform(0, 1)
        if np.log(u) < llh_new - llh:
            x = x_new
            llh = llh_new

        return x, llh