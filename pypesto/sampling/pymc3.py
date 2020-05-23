import numpy as np
from typing import Union
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
    import arviz as az
except ImportError:
    pass


class Pymc3Sampler(Sampler):
    """Wrapper around Pymc3 samplers.

    Parameters
    ----------
    step_function:
        A pymc3 step function, e.g. NUTS, Slice. If not specified, pymc3
        determines one automatically (preferable).
    **options:
        Options are directly passed on to `pymc3.sample`.
    """

    def __init__(self, step_function=None, **options):
        super().__init__(options)
        self.step_function = step_function
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.Text, None] = None
        self.data: Union[az.InferenceData, None] = None

    @classmethod
    def translate_options(cls, options):
        if options is None:
            options = {}
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        self.x0 = x0
        self.trace = None
        self.data = None

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

        # create model context
        with pm.Model() as model:
            # uniform prior
            k = [pm.Uniform(x_name, lower=lb, upper=ub)
                 for x_name, lb, ub in
                 zip(problem.x_names, problem.lb, problem.ub)]

            # convert to tensor vector
            theta = tt.as_tensor_variable(k)

            # use a DensityDist (use a lambda function to "call" the Op)
            pm.DensityDist('llh', logp=lambda v: llh(v),
                           observed={'v': theta})

            # step, by default automatically determined by pymc3
            step = None
            if self.step_function:
                step = self.step_function()

            # perform the actual sampling
            trace = pm.sample(
                draws=int(n_samples), trace=trace, start=x0, step=step,
                **self.options)

            # convert trace to inference data object
            data = az.from_pymc3(trace=trace, model=model)

        self.trace = trace
        self.data = data

    def get_samples(self) -> McmcPtResult:
        # parameter values
        trace_x = np.asarray(
            self.data.posterior.to_array()).transpose((1, 2, 0))

        # TODO this is only the log-likelihood
        trace_fval = np.asarray(self.data.log_likelihood.to_array())
        # remove trailing dimensions
        trace_fval = np.reshape(trace_fval, trace_fval.shape[1:-1])
        # flip sign
        trace_fval = - trace_fval

        if trace_x.shape[0] != trace_fval.shape[0] \
                or trace_x.shape[1] != trace_fval.shape[1] \
                or trace_x.shape[2] != len(self.problem.x_names):
            raise ValueError("Trace dimensions are inconsistent")

        return McmcPtResult(
            trace_x=np.array(trace_x),
            trace_fval=np.array(trace_fval),
            betas=np.array([1.] * trace_x.shape[0]),
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
