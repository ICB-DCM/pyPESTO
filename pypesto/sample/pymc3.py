import numpy as np
from typing import Dict, Union
import logging

from ..objective import Objective, History
from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult

logger = logging.getLogger(__name__)

try:
    import pymc3 as pm
    import theano.tensor as tt
    from theano.gof.null_type import NullType
except ImportError:
    pass


class Pymc3Sampler(Sampler):
    """Wrapper around Pymc3 samplers."""

    def __init__(self, options: Dict = None):
        super().__init__(options)
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.Text, None] = None

    @classmethod
    def translate_options(cls, options):
        if options is None:
            options = {}
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        self.x0 = x0
        self.trace = None

        self.problem.objective.history = History()

    def sample(
            self, n_samples: int, beta: float = 1.
    ):
        problem = self.problem
        llh = TheanoLogLikelihood(problem, beta)
        trace = self.trace

        x0 = None
        if self.x0 is not None:
            x0 = {x_name: val
                  for x_name, val in zip(self.problem.x_names, self.x0)}

        # use PyMC3 to draw samples
        with pm.Model():
            # uniform prior
            k = [pm.Uniform(x_name, lower=lb, upper=ub)
                 for x_name, lb, ub in
                 zip(problem.x_names, problem.lb, problem.ub)]

            # convert m and c to a tensor vector
            theta = tt.as_tensor_variable(k)

            # use a DensityDist (use a lambda function to "call" the Op)
            pm.DensityDist('likelihood', lambda v: llh(v),
                           observed={'v': theta})

            trace = pm.sample(
                draws=int(n_samples), trace=trace, start=x0, **self.options)

        self.trace = trace

    def get_samples(self) -> McmcPtResult:
        trace_x = np.array(
            [self.trace.get_values(x_name)
             for x_name in self.problem.x_names]
        ).T

        # TOOD
        trace_fval = np.empty(trace_x.shape[0])
        trace_fval[:] = np.nan

        return McmcPtResult(
            trace_x=np.array([trace_x]),
            trace_fval=np.array([trace_fval]),
            betas=np.array([1.]),
        )


class TheanoLogLikelihood(tt.Op):
    """
    Theano wrapper around the log-likelihood function.
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective

        # initialize the llh Op
        self._llh = lambda x: - beta * self._objective(x, sensi_orders=(0,))

        # initialize the sllh Op
        if problem.objective.has_grad:
            self._sllh = TheanoLogLikelihoodGradient(problem, beta)
        else:
            self._sllh = None

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        llh = self._llh(theta)
        outputs[0][0] = np.array(llh)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        if self._sllh is None:
            return [NullType]
        theta, = inputs
        sllh = self._sllh(theta)
        return [g[0] * sllh]


class TheanoLogLikelihoodGradient(tt.Op):
    """
    Theano wrapper around the log-likelihood gradient function.
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective
        self._sllh = lambda x: - beta * self._objective(x, sensi_orders=(1,))

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate gradients
        sllh = self._sllh(theta)
        outputs[0][0] = sllh
