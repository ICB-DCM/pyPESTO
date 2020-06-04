import math
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
    from .pymc3_model import create_pymc3_model, pymc3_to_arviz
except ImportError:
    pm = None


class Pymc3Sampler(Sampler):
    """Wrapper around Pymc3 samplers.

    Parameters
    ----------
    step_function:
        A pymc3 step function, e.g. NUTS, Slice. If not specified, pymc3
        determines one automatically (preferable).
    cache_gradients:
        If `True`, evaluate gradient together with objective value.
        This results in a modest speed-up for `pymc3` samplers using gradient
        information (such as NUTS).
        It can be safely kept on even when using other samplers.
    vectorize:
        If `True`, all free parameters will be gathered in a single vector
        inside the `pymc3.Model` object. This results in a small speed-up,
        especially if the objective computation time is small or if simpler
        step functions (such as `pymc3.Metropolis`) are used.
    **kwargs:
        Options are directly passed on to `pymc3.sample`.
    """

    def __init__(self, step_function=None, cache_gradients: bool = True,
                 vectorize: bool = True, **kwargs):
        if pm is None:
            raise Exception('Please install the pymc3 package ' \
                            'in order to use the Pymc3Sampler sampler.')
        super().__init__(kwargs)
        self.step_function = step_function
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.base.MultiTrace, None] = None
        self.data: Union[az.InferenceData, None] = None
        # pymc3 model options
        self.cache_gradients = cache_gradients
        self.vectorize = vectorize

    @classmethod
    def translate_options(cls, options):
        if not options:
            options = {}
        options.setdefault('chains', 1)
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        self.x0 = x0
        self.trace = None
        self.data = None

        self.problem.objective.history = History()

    def sample(self, n_samples: int, beta: float = 1.):
        # Create PyMC3 model
        model = create_pymc3_model(self.problem, self.x0, beta,
                                   cache_gradients=self.cache_gradients,
                                   vectorize=self.vectorize)

        # Check posterior at starting point
        if self.trace is None:
            logps = [RV.logp(model.test_point) for RV in model.basic_RVs]
            if not all(math.isfinite(logp) for logp in logps):
                raise Exception('Log-posterior of same basic random variables' \
                                ' is not finite. Please report this issue at ' \
                                'https://github.com/ICB-DCM/pyPESTO/issues' \
                                '\nLog-posterior at test point is\n' + \
                                str(model.check_test_point()))

        with model:
            # step, by default automatically determined by pymc3
            step = None
            if self.step_function:
                step = self.step_function()

            # select start point
            if self.trace is None:
                start = None
                # NB the start point will be based on the model test point,
                #    but may be modified if NUTS is auto-assigned (e.g. jitter).
                #    If we want to be sure that the test point is exaclty used,
                #    the init='adapt_diag' or init='adapt_full' should be used
            else:
                raise NotImplementedError('resuming a pymc3 chain '
                                          'is currently not implemented')
                # TODO to implement this case, we should copy the auto-assign
                #      code from pymc3 inside pypesto, so that we have a
                #      reference to the step_method that gets tuned

            # Sample from model
            trace = pm.sample(draws=int(n_samples), start=start, step=step,
                              trace=self.trace, **self.options)

            # NB in theory we could just pass model as a keyword argument
            #    to both sample and step_function, but if step_function is None
            #    and NUTS cannot be assigned, the creation of a new default
            #    step method fails because there is no active model
            #    (probably a bug in pymc 3.8)

        # Convert trace to inference data object
        data = pymc3_to_arviz(model, trace)

        self.trace = trace
        self.data = data

    def get_samples(self) -> McmcPtResult:
        # parameter values
        trace_x = np.asarray(self.data.posterior.to_array())
        if self.vectorize and len(self.problem.x_free_indices) > 1:
            # array dimensions are ordered as
            # (variable, chain, draw, variable coordinates)
            assert trace_x.shape[0] == 1  # all free variables have been packed
                                          # in a single vector variable
            trace_x = np.squeeze(trace_x, axis=0)
        else:
            # array dimensions are ordered as
            # (variable, chain, draw)
            trace_x = trace_x.transpose((1, 2, 0))

        # NB samplers like AdaptiveMetropolisSampler include the starting point
        #    in the trace. Since pymc3 has a tuning process, it makes no sense
        #    to include the starting point in this case

        # Since the priors in the pymc3 model are artificial
        # and since pypesto objective include the real prior,
        # the log-likelihood of the pymc3 model
        # is actually the real model's log-posterior
        # (i.e., the negative objective value)
        # TODO this is only the negative objective values
        trace_fval = np.asarray(self.data.log_likelihood.to_array())
        # remove trailing dimensions
        trace_fval = np.reshape(trace_fval, trace_fval.shape[1:-1])
        # flip sign
        trace_fval = - trace_fval

        if trace_x.shape[0] != trace_fval.shape[0] \
                or trace_x.shape[1] != trace_fval.shape[1] \
                or trace_x.shape[2] != len(self.problem.x_free_indices):
            raise ValueError("Trace dimensions are inconsistent")

        return McmcPtResult(
            trace_x=np.array(trace_x),
            trace_fval=np.array(trace_fval),
            betas=np.array([1.] * trace_x.shape[0]),
        )
