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
    import arviz as az
    import theano.tensor as tt
    from theano.ifelse import ifelse
except ImportError:
    pm = az = tt = ifelse = None

try:
    from .theano import TheanoLogProbability, CachedObjective
except (AttributeError, ImportError):
    TheanoLogProbability = None
    CachedObjective = None


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

    def __init__(self, step_function=None, cache_gradients: bool = True, **kwargs):
        super().__init__(kwargs)
        self.step_function = step_function
        self.cache_gradients = cache_gradients
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pm.backends.base.MultiTrace, None] = None
        self.data: Union[az.InferenceData, None] = None

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
        # step, by default automatically determined by pymc3
        step = None
        if self.step_function:
            step = self.step_function()

        # Create PyMC3 model
        model = PyMC3Model(self.problem, self.x0, beta,
                           cache_gradients=self.cache_gradients)

        # Check posterior at starting point
        if self.trace is None:
            logps = [RV.logp(model.test_point) for RV in model.basic_RVs]
            if not all(math.isfinite(logp) for logp in logps):
                raise Exception('Log-posterior of same basic random variables' \
                                ' is not finite. Please report this issue at ' \
                                'https://github.com/ICB-DCM/pyPESTO/issues' \
                                '\nLog-posterior at test point is\n' + \
                                str(model.check_test_point()))

        # Sample from mmodel
        trace = pm.sample(draws=int(n_samples), start=None, step=step,
                          trace=self.trace, model=model, **self.options)

        # Convert trace to inference data object
        data = pymc3_to_arviz(model, trace)

        self.trace = trace
        self.data = data

    def get_samples(self) -> McmcPtResult:
        # parameter values
        trace_x = np.asarray(
            self.data.posterior.to_array()).transpose((1, 2, 0))

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


def create_pymc3_model(problem: Problem,
                       testval: Union[np.ndarray, None] = None,
                       beta: float = 1., *,
                       cache_gradients: bool = True,
                       vectorize: bool = False,
                       lerp: str = 'convex',
                       verbose: bool = False):
        with pm.Model() as model:
            # Wrap objective in a theno op (applying caching if needed)
            objective = problem.objective
            if objective.has_grad and cache_gradients:
                objective = CachedObjective(objective)
            log_post_fun = TheanoLogProbability(objective, beta)

            # If a test value is given, check its size
            x_free_names = [problem.x_names[i] for i in problem.x_free_indices]
            if testval is not None and len(testval) != len(x_free_names):
                raise ValueError('The size of the test value must be equal ' \
                                 'to the number of free parameters')

            # If a test value is given, correct values at the optimization
            # boundaries moving them just inside the interval.
            # This is due to the fact that pymc3 maps bounded variables
            # to the whole real line.
            # see issue #365 at https://github.com/ICB-DCM/pyPESTO/issues/365
            if testval is not None:
                for i in range(len(x_free_names)):
                    lb, ub = problem.lb[i], problem.ub[i]
                    x = testval[i]
                    if lb < x < ub:
                        # Inside bounds, OK
                        pass
                    elif x < lb or x > ub:
                        raise ValueError(f'testval[{i}] is out of bounds')
                    else:
                        # Move this parameter inside the interval
                        # by taking the nearest floating point value
                        # (it appears this is enough to solve the problem)
                        if x == lb:
                            testval[i] = np.nextafter(lb, ub)
                        else:
                            assert x == ub
                            testval[i] = np.nextafter(ub, lb)

            # Create a uniform bounded variable for each parameter
            if vectorize:
                if testval is None:
                    theta = BetterUniform(pymc3_vector_parname(problem),
                                          lerp=lerp,
                                          lower=problem.lb, upper=problem.ub)
                else:
                    theta = BetterUniform(pymc3_vector_parname(problem),
                                          lerp=lerp, testval=testval,
                                          lower=problem.lb, upper=problem.ub)
            elif testval is None:
                k = [BetterUniform(x_name, lower=lb, upper=ub, lerp=lerp)
                         for x_name, lb, ub in
                         zip(x_free_names, problem.lb, problem.ub)]
            else:
                k = [BetterUniform(x_name, testval=x, lower=lb, upper=ub, lerp=lerp)
                         for x_name, x, lb, ub in
                         zip(x_free_names, testval, problem.lb, problem.ub)]

            # Convert to tensor vector
            if not vectorize:
                theta = tt.as_tensor_variable(k)

            # Use a DensityDist for the log-posterior
            pm.DensityDist('log_post', logp=log_post_fun, observed=theta)

        if verbose:
            print('Evaluating log-posterior at test point')
            print(model.check_test_point())

        return model


def pymc3_vector_parname(problem: Problem):
    if 'theta' in problem.x_names:
        if 'free_parameters' in problem.x_names:
            if 'pymc3_free_parameters' in problem.x_names:
                raise Exception('cannot find a name for the compacted parameters')
            else:
                return 'pymc3_free_parameters'
        else:
            return 'free_parameters'
    else:
        return 'theta'


def pymc3_to_arviz(model: pm.Model, trace: pm.backends.base.MultiTrace, problem: Problem = None, vectorize: bool = False):
    kwargs = {}
    if vectorize:
        if problem is None:
            raise ValueError('if vectorize is True, ' \
                             'a problem must be given')
        x_free_names = [problem.x_names[i] for i in problem.x_free_indices]
        kwargs['coords'] = {"free_parameter": x_free_names}
        kwargs['dims'] = {pymc3_vector_parname(problem): ["free_parameter"]}

    return az.from_pymc3(
        trace=trace,
        model=model,
        log_likelihood=False,  # NB at the moment we do not separate prior
                               # from posterior, thus computing likelihood
                               # from pymc3 model is meaningless
        **kwargs
    )


def BetterUniform(name, *, lower, upper, lerp='convex', **kwargs):
    """
    A uniform bounded random variable whose behaviour near the boundary of
    the domain is better than the native `pymc3.Uniform`.

    The problem with `pymc3.Uniform` is that the inverse transform formula
    fails in floating point arithmetic when the lower and the upper bound differ
    largely in magnitude, resulting in a value outside the original
    interval. This leads the log-posterior for the original distribution to
    become `-inf`. By using a better formula this can be avoided.

    While this is a sufficient fix, there is another opportunity for
    simplification: avoiding the computation of `log(ub - lb)` in the
    log-posterior (it appears both in the original log-posterior and in the
    transform jacobian, so it can be simplified away).
    This simplification is imposed by using instead of `Uniform` a `Flat` prior
    (which has log-posterior 0) and removing the term `log(ub - lb)` from
    the interval transformation jacobian.
    """
    if 'shape' not in kwargs.keys():
        if 'testval' in kwargs.keys():
            shape = np.broadcast(lower, upper, kwargs['testval']).shape
        else:
            shape = np.broadcast(lower, upper).shape
        kwargs['shape'] = shape
    scalar = (len(kwargs['shape']) == 0)

    if 'transform' in kwargs.keys():
        raise Exception('if specifying a custom transform, ' \
                        'please use pymc3.Uniform')

    BoundedFlat = pm.Bound(pm.Flat, lower=lower, upper=upper)

    transform = BetterInterval(lower, upper, lerp, scalar=scalar)

    return BoundedFlat(name, transform=transform, **kwargs)

    # In the case we start from pm.Uniform,
    # we need to comment the jacobian out of BetterInterval
    # transform = BetterInterval(lower, upper, lerp, scalar=scalar)
    # return pm.Uniform(name, lower=lower, upper=upper, transform=transform, **kwargs)


class BetterInterval(pm.distributions.transforms.Interval):
    name = "betterinterval"
    def __init__(self, a, b, lerp, *, scalar=True):
        super().__init__(a, b)

        scalar_bounds = isinstance(a, float) and isinstance(b, float)
        assert scalar_bounds or not scalar

        if not scalar:
            allowed_lerp = ('convex', 'clipped', 'auto') if scalar_bounds \
                           else ('convex', 'clipped')
            if lerp not in allowed_lerp:
                raise NotImplementedError(f'lerp method {lerp} is unsupported' \
                                           ' when using a vector of parameters')

        if lerp == 'auto':
            # Could possibly be improved by comparing orders of magnitude
            if (a <= 0 and b >= 0) or (a >= 0 and b <= 0):
                lerp = 'convex'
            else:
                lerp = 'clipped'

        self.lerp_method = lerp

    def backward(self, x):
        a, b = self.a, self.b
        f = tt.nnet.sigmoid(x)
        return self.lerp(f, a, b)

    if pm.__version__ == '3.8':
        def backward_val(self, x):
            a, b = self.a_, self.b_
            f = 1 / (1 + np.exp(-x))
            return self.lerp(f, a, b)

    def jacobian_det(self, x):
        s = tt.nnet.softplus(-x)
        return -2 * s - x

    def lerp(self, f, a, b):
        if self.lerp_method == 'convex':
            return f * b + (1 - f) * a
            # Fastest, but slighlty less precise in common situations
        elif self.lerp_method == 'clipped':
            return tt.minimum(a + f * (b - a), b)
            # A bit more precise, except when a and b have different sign
            # See lerp == 'auto'
        elif self.lerp_method == 'piecewise':
            return ifelse(f < 0.5, a + f * (b - a), b - (1 - f) * (b - a))
            # Better than the clipped formula,
            # except when a and b have different sign
            # Not sure if when computing the gradient,
            # theano can simplify the branch
        else:
            raise ValueError(f'unknown lerp method {self.lerp_method}')
