from typing import Union, Tuple, List, Optional
import numpy as np

import pymc3 as pm
import arviz as az
import theano.tensor as tt
from theano.ifelse import ifelse

from ..problem import Problem
from .theano import TheanoLogProbability, CachedObjective


def create_pymc3_model(problem: Problem,
                       testval: Optional[np.ndarray] = None,
                       beta: float = 1., *,
                       support: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       jitter_scales: Optional[np.ndarray] = None,
                       cache_gradients: bool = True,
                       vectorize: bool = True,
                       lerp_method: str = 'convex',
                       verbose: bool = False):
        # Check consistency of arguments
        if jitter_scales is not None and testval is None:
            raise ValueError('if testval is not given, '
                             'jitter_scales cannot be given')

        # Extract the names of the free parameters
        x_names = [problem.x_names[i] for i in problem.x_free_indices]
        assert len(x_names) == problem.dim

        # If there is only one free parameter, then we should not vectorize
        if problem.dim == 0:
            raise Exception('Cannot sample: no free parameters')
        elif problem.dim == 1:
            vectorize = False

        # Overwrite lower and upper bounds
        if support is not None:
            lbs, ubs = support
            if len(lbs) != problem.dim or len(ubs) != problem.dim:
                raise ValueError('length of lower/upper bounds '
                                 'must be equal to number of free parameters')
        else:
            lbs, ubs = problem.lb, problem.ub

        # Disable vectorize if there are non-finite bounds
        # TODO implement vectorize for this case too
        if np.isinf(lbs).any() or np.isinf(ubs).any():
            vectorize = False

        with pm.Model() as model:
            # Wrap objective in a theno op (applying caching if needed)
            objective = problem.objective
            if objective.has_grad and cache_gradients:
                objective = CachedObjective(objective)
            log_post_fun = TheanoLogProbability(objective, beta)

            # If a test value is given, check its size
            if testval is not None and len(testval) != problem.dim:
                raise ValueError('The size of the test value must be equal ' \
                                 'to the number of free parameters')

            # If a test value is given, correct values at the optimization
            # boundaries moving them just inside the interval.
            # This is due to the fact that pymc3 maps bounded variables
            # to the whole real line.
            # see issue #365 at https://github.com/ICB-DCM/pyPESTO/issues/365
            if testval is not None:
                for i in range(problem.dim):
                    lb, ub = lbs[i], ubs[i]
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
                    theta = BetterUniform(pymc3_vector_parname(x_names),
                                          lower=lbs, upper=ubs,
                                          lerp_method=lerp_method)
                else:
                    theta = BetterUniform(pymc3_vector_parname(x_names),
                                          lower=lbs, upper=ubs,
                                          testval=testval,
                                          jitter_scale=jitter_scales,
                                          lerp_method=lerp_method)
            elif testval is None:
                k = [BetterUniform(x_name, lower=lb, upper=ub,
                                   lerp_method=lerp_method)
                         for x_name, lb, ub in
                         zip(x_names, lbs, ubs)]
            elif jitter_scales is None:
                k = [BetterUniform(x_name, lower=lb, upper=ub, testval=x,
                                   lerp_method=lerp_method)
                         for x_name, x, lb, ub in
                         zip(x_names, testval, lbs, ubs)]
            else:
                k = [BetterUniform(x_name, lower=lb, upper=ub, testval=x,
                                   jitter_scale=jitter_scale,
                                   lerp_method=lerp_method)
                         for x_name, x, lb, ub, jitter_scale in
                         zip(x_names, testval, lbs, ubs, jitter_scales)]

            # Convert to tensor vector
            if not vectorize:
                theta = tt.as_tensor_variable(k)

            # Use a DensityDist for the log-posterior
            pm.DensityDist(pymc3_logp_parname(x_names),
                           logp=log_post_fun, observed=theta)

        if verbose:
            print('Evaluating log-posterior at test point')
            print(model.check_test_point())

        return model


def pymc3_vector_parname(x_names: List[str]):
    if 'theta' in x_names:
        if 'free_parameters' in x_names:
            if 'pymc3_free_parameters' in x_names:
                raise Exception('cannot find a name for the parameter vector')
            else:
                return 'pymc3_free_parameters'
        else:
            return 'free_parameters'
    else:
        return 'theta'


def pymc3_logp_parname(x_names: List[str]):
    if 'log_post' in x_names:
        if 'pymc3_log_post' in x_names:
            raise Exception('cannot find a name for the log-posterior')
        else:
            return 'pymc3_log_post'
    else:
        return 'log_post'


def pymc3_to_arviz(model: pm.Model, trace: pm.backends.base.MultiTrace, *,
                   problem: Problem = None, x_names: List[str] = None):
    if problem is not None and x_names is not None:
        raise ValueError('the problem and x_names keyword arguments '
                         'cannot both be given.')

    # Determine if free parameters were vectorized
    kwargs = {}
    if len(model.free_RVs) == 1:
        theta = model.free_RVs[0]
        if len(theta.distribution.shape) > 0:
            # theta is a tensor variable
            assert len(theta.distribution.shape) == 1  # should only be vector
            if problem is None and x_names is None:
                raise ValueError('if vectorize is True, one of the problem '
                                 'or x_names keyword arguments must be given.')
            if x_names is None:
                x_names = [problem.x_names[i] for i in problem.x_free_indices]
            kwargs['coords'] = {"free_parameter": x_names}
            kwargs['dims'] = {pymc3_vector_parname(x_names): ["free_parameter"]}

    return az.from_pymc3(
        trace=trace,
        model=model,
        log_likelihood=True,  # pymc3 log-likelihood == negative obj. value
                              #                      == real model's posterior
        **kwargs
    )


def BetterUniform(name, *, lower, upper, jitter_scale=None,
                  lerp_method='convex', **kwargs):
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
    lower, upper = np.asarray(lower), np.asarray(upper)

    if 'shape' not in kwargs.keys():
        # Derive the shape of the random variable by broadcast
        if 'testval' in kwargs.keys():
            shape = np.broadcast(lower, upper, kwargs['testval']).shape
        else:
            shape = np.broadcast(lower, upper).shape
        kwargs['shape'] = shape
    scalar = (len(kwargs['shape']) == 0)

    if 'transform' in kwargs.keys():
        raise Exception('if specifying a custom transform, ' \
                        'please use pymc3.Uniform')

    if jitter_scale is not None and kwargs.get('testval', None) is None:
            raise ValueError('if testval is not given, '
                             'jitter_scale cannot be given')

    # Check bounds ordering
    if not (lower <= upper).all():
        raise ValueError('each lower bound should be smaller than '
                         'the corresponding upper bound')

    # Depending on the bounds, choose the transformation
    if np.isfinite(lower).all() and np.isfinite(upper).all():
        BoundedFlat = pm.Bound(pm.Flat, lower=lower, upper=upper)
        transform = BetterInterval(lower, upper,
                                   testval=kwargs.get('testval', None),
                                   jitter_scale=jitter_scale,
                                   lerp_method=lerp_method, scalar=scalar)
        return BoundedFlat(name, transform=transform, **kwargs)
        # In the case we start from pm.Uniform,
        # we need to comment the jacobian out of BetterInterval
        # transform = BetterInterval(lower, upper,
        #                            lerp_method=lerp_method, scalar=scalar)
        # return pm.Uniform(name, lower=lower, upper=upper,
        #                   transform=transform, **kwargs)
    elif jitter_scale is not None:
        raise NotImplementedError('jitter_scale for unbounded supports '
                                  'has not been implemented yet.')
    elif shape == ():
        if lower == -np.inf:
            if upper == np.inf:
                return pm.Flat(name, **kwargs)
            else:
                BoundedFlat = pm.Bound(pm.Flat, upper=upper)
                return BoundedFlat(name, **kwargs)
        else:
            assert upper == np.inf
            if lower == 0:
                return pm.HalfFlat(name, **kwargs)
            else:
                BoundedFlat = pm.Bound(pm.Flat, lower=lower)
                return BoundedFlat(name, **kwargs)
    else:
        raise NotImplementedError('in the non-scalar case, unbounded supports '
                                  'have not been implemented yet.')

def is_pymc3_version_newer_than_3_8():
    return hasattr(pm.parallel_sampling, 'progress_bar')

class BetterInterval(pm.distributions.transforms.Interval):
    name = "betterinterval"
    def __init__(self, a, b, *,
                 jitter_scale=None, testval=None,
                 lerp_method='convex', scalar=True):
        super().__init__(a, b)

        scalar_bounds = np.shape(a) == () and np.shape(b) == ()
        assert scalar_bounds or not scalar

        if not scalar:
            allowed_lerp = ('convex', 'clipped', 'auto') if scalar_bounds \
                           else ('convex', 'clipped')
            if lerp_method not in allowed_lerp:
                raise NotImplementedError(f'lerp method {lerp_method} is ' \
                                           'unsupported when using a vector ' \
                                           'of parameters')

        if lerp_method == 'auto':
            # Could possibly be improved by comparing orders of magnitude
            if (a <= 0 and b >= 0) or (a >= 0 and b <= 0):
                lerp_method = 'convex'
            else:
                lerp_method = 'clipped'

        if jitter_scale is not None and testval is None:
            raise ValueError('if testval is not given, '
                             'jitter_scale cannot be given')

        if not is_pymc3_version_newer_than_3_8():
            self._alpha = BetterInterval.compute_alpha(self.a_, self.b_,
                                                       testval, jitter_scale)

        self.alpha = BetterInterval.compute_alpha(self.a, self.b,
                                                  testval, jitter_scale)

        self.lerp_method = lerp_method

    @staticmethod
    def compute_alpha(a, b, testval, jitter_scale, alphamin=1e-6):
        if jitter_scale is None:
            return None
        else:
            if not isinstance(a, np.ndarray):
                testval = tt.as_tensor_variable(testval)
                jitter_scale = tt.as_tensor_variable(jitter_scale)
            alpha = (testval - a) * (b - testval) / (jitter_scale * (b - a))
            if alphamin > 0:
                # NB if testval is very close to the boundary,
                #    alpha may become very small, which can lead
                #    to -inf log-probabilities or tuning problems.
                #    For this reason it is best to choose a sensible lower limit
                #    TODO which lower limit is sensible?
                if isinstance(a, np.ndarray):
                    return np.maximum(alpha, alphamin)
                else:
                    alphamin = tt.as_tensor_variable(alphamin)
                    return tt.maximum(alpha, alphamin)
            else:
                return alpha

    def forward(self, x):
        y = super().forward(x)
        return y if self.alpha is None else self.alpha * y

    def forward_val(self, x, point=None):
        y = super().forward_val(x, point)
        if self._alpha is None:
            return y
        else:
            return np.multiply(self._alpha, y, dtype=y.dtype)

    def backward(self, x):
        a, b = self.a, self.b
        if self.alpha is not None:
            x = x / self.alpha
        f = tt.nnet.sigmoid(x)
        return self.lerp(f, a, b)

    if not is_pymc3_version_newer_than_3_8():
        def backward_val(self, x):
            a, b = self.a_, self.b_
            if self._alpha is not None:
                x = np.asarray(x)
                x = np.divide(x, self._alpha, dtype=x.dtype)
            f = 1 / (1 + np.exp(-x))
            return self.lerp(f, a, b)

    def jacobian_det(self, x):
        if self.alpha is not None:
            x = x / self.alpha
        s = tt.nnet.softplus(-x)
        J = -2 * s - x
        if self.alpha is not None:
            J = J - tt.log(self.alpha)
        return J

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
