import numpy as np
from typing import Union, Optional, Dict
import logging

from ..objective import History
from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult

logger = logging.getLogger(__name__)

try:
    import pymc3 as pm
    from .pymc3_interface import (
        create_pymc3_model,
        pymc3_to_arviz,
        arviz_to_pypesto,
        filter_create_step_method_kwargs,
        create_step_method,
        init_random_seed
    )
    import arviz as az
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
    remap_to_reals:
        If `True`, parameters will be remapped to the whole real line for
        use in PyMC3 (this is required by the NUTS sampler). If not specified,
        it will be set to `True` if NUTS is used and to `False` otherwise.
    **kwargs:
        Options are directly passed on to `pymc3.sample`.
    """

    def __init__(self, cache_gradients: bool = True,
                       vectorize: bool = True,
                       remap_to_reals: Optional[bool] = None, **kwargs):

        if pm is None:
            raise Exception('Please install the pymc3 package ' \
                            'in order to use the Pymc3Sampler sampler.')

        super().__init__(kwargs)

        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.base.MultiTrace, None] = None
        self.data: Union[az.InferenceData, None] = None

        self.cache_gradients = cache_gradients
        self.vectorize = vectorize
        self.remap_to_reals = remap_to_reals

    @classmethod
    def translate_options(cls, options):
        if not options:
            options = {}
        options.setdefault('chains', 1)
        # No jittering by default:
        # safer in the case the objective fails often
        # for points far from the starting one
        options.setdefault('init', 'adapt_diag')
        options.setdefault('tune', 1000)  # as pymc3
        options.setdefault('discard_tuned_samples', True)  # as pymc3
        # Initialize random seed, so that the same seed is used both
        # for creating the step_method and the sampling
        options['random_seed'] = init_random_seed(
            options.get('random_seed', None), options['chains']
        )
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        self.problem = problem
        if x0 is not None:
            if len(x0) == problem.dim:
                x0 = np.asarray(x0)
            else:
                x0 = problem.get_reduced_vector(x0)
        self.x0 = x0
        self.trace = None
        self.data = None

        self.problem.objective.history = History()

    def sample(self, n_samples: int, beta: float = 1.):
        if self.trace is not None:
            raise Exception('PyMC3 sampling cannot be resumed '
                            'using the pyPESTO interface. '
                            'Consider using the lower level interface '
                            'in pypesto.sample.pymc3_interface.')

        # Create PyMC3 model
        remap_to_reals = True if self.remap_to_reals is None \
                         else self.remap_to_reals
        model = create_pymc3_model(self.problem, self.x0, beta,
                                   cache_gradients=self.cache_gradients,
                                   vectorize=self.vectorize,
                                   remap_to_reals=remap_to_reals)

        # Create the step method
        step_kwargs = filter_create_step_method_kwargs(self.options)
        step, start = create_step_method(model, **step_kwargs)
        # NB the start point will be based on the model test point,
        #    but may be modified if NUTS is auto-assigned (e.g. by jitter).
        #    If we want to be sure that the test point is exactly used,
        #    the init='adapt_diag' or init='adapt_full' should be used

        # Automatic choice of remap_to_reals
        if not isinstance(step, pm.NUTS):
            if __debug__:
                if isinstance(step, pm.step_methods.CompoundStep):
                    assert all(not isinstance(m, pm.NUTS) for m in step.methods)
                else:
                    assert isinstance(step,
                                      pm.step_methods.arraystep.BlockedStep)
            if self.remap_to_reals is None:
                # Rebuild problem with remap_to_reals = False
                model = create_pymc3_model(self.problem, self.x0, beta,
                                           cache_gradients=self.cache_gradients,
                                           vectorize=self.vectorize,
                                           remap_to_reals=False)
                # Rebuild stepper (exclude NUTS so that no messages are printed)
                step, start = create_step_method(model,
                                                 **dict(step_kwargs, init=None))

        # Keyword arguments for sampling functions
        sample_kwargs = self.options.copy()
        sample_kwargs['step'] = step

        # Sampling
        draws = int(n_samples)
        with model:
            trace = pm.sample(draws=draws, start=start, **sample_kwargs)

        # Convert trace to inference data object
        data = pymc3_to_arviz(model, trace,
                              problem=self.problem, save_warmup=True)

        self.trace = trace
        self.data = data

    def get_samples(self) -> McmcPtResult:
        if self.options['discard_tuned_samples'] and self.options['tune'] > 0:
            burn_in = 0  # Tuning samples have been drawn and discarded
        else:
            burn_in = 'auto'  # Determine burn-in from warm-up data (if present)
        return arviz_to_pypesto(self.problem, self.data,
                                save_warmup=True, burn_in=burn_in, full=False)
