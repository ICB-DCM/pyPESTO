import numpy as np
from typing import Union
import logging

from ..objective import History
from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult

logger = logging.getLogger(__name__)

try:
    import pymc3 as pm
    import arviz as az
    import theano.tensor as tt
except ImportError:
    pm = az = tt = None

try:
    from .theano import TheanoLogProbability
except (AttributeError, ImportError):
    TheanoLogProbability = None


class Pymc3Sampler(Sampler):
    """Wrapper around Pymc3 samplers.

    Parameters
    ----------
    step_function:
        A pymc3 step function, e.g. NUTS, Slice. If not specified, pymc3
        determines one automatically (preferable).
    **kwargs:
        Options are directly passed on to `pymc3.sample`.
    """

    def __init__(self, step_function=None, **kwargs):
        super().__init__(kwargs)
        self.step_function = step_function
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.Text, None] = None
        self.data: Union[az.InferenceData, None] = None

    @classmethod
    def translate_options(cls, options):
        if not options:
            options = {'chains': 1}
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        if x0 is not None:
            if len(x0) != problem.dim:
                x0 = problem.get_reduced_vector(x0)
        self.x0 = x0
        self.trace = None
        self.data = None

        self.problem.objective.history = History()

    def sample(
            self, n_samples: int, beta: float = 1.):
        problem = self.problem
        log_post_fun = TheanoLogProbability(problem, beta)
        trace = self.trace

        x0 = None
        if self.x0 is not None:
            x0 = {x_name: val
                  for x_name, val in zip(self.problem.x_names, self.x0)}

        # create model context
        with pm.Model() as model:
            # uniform bounds
            k = [pm.Uniform(x_name, lower=lb, upper=ub)
                 for x_name, lb, ub in
                 zip(problem.get_reduced_vector(problem.x_names),
                     problem.lb, problem.ub)]

            # convert to tensor vector
            theta = tt.as_tensor_variable(k)

            # use a DensityDist for the log-posterior
            pm.DensityDist('log_post', logp=lambda v: log_post_fun(v),
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

        # TODO this is only the negative objective values
        trace_neglogpost = np.asarray(self.data.log_likelihood.to_array())
        # remove trailing dimensions
        trace_neglogpost = np.reshape(trace_neglogpost,
                                      trace_neglogpost.shape[1:-1])
        # flip sign
        trace_neglogpost = - trace_neglogpost

        if trace_x.shape[0] != trace_neglogpost.shape[0] \
                or trace_x.shape[1] != trace_neglogpost.shape[1] \
                or trace_x.shape[2] != self.problem.dim:
            raise ValueError("Trace dimensions are inconsistent")

        return McmcPtResult(
            trace_x=np.array(trace_x),
            trace_neglogpost=np.array(trace_neglogpost),
            trace_neglogprior=np.full(trace_neglogpost.shape, np.nan),
            betas=np.array([1.] * trace_x.shape[0]),
        )
