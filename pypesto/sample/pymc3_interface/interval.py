"""
Improved transformation from a bounded interval to the whole real line
(needed for using the NUTS sampler).
Compared to pymc3.Interval, it allows to set a scale for the transformation,
so that the jitter around the test value applied by pymc3.init_nuts
can be controlled.
"""

import numpy as np

import pymc3 as pm
from pymc3.theanof import floatX

import theano.tensor as tt
from theano.ifelse import ifelse


def _variable_shape(lower, upper, testval=None):
    if testval is None:
        return np.broadcast(lower, upper).shape
    else:
        return np.broadcast(lower, upper, testval).shape


def ScaleAwareUniform(name, *, lower, upper, jitter_scale=None,
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

    kwargs.setdefault('shape', _variable_shape(lower, upper,
                                               kwargs.get('testval', None)))
    scalar = kwargs['shape'] == ()

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
        transform = ScaleAwareInterval(lower, upper,
                                   testval=kwargs.get('testval', None),
                                   jitter_scale=jitter_scale,
                                   lerp_method=lerp_method, scalar=scalar)
        return BoundedFlat(name, transform=transform, **kwargs)
        # In the case we start from pm.Uniform,
        # we need to comment the jacobian out of ScaleAwareInterval
        # transform = ScaleAwareInterval(lower, upper,
        #                            lerp_method=lerp_method, scalar=scalar)
        # return pm.Uniform(name, lower=lower, upper=upper,
        #                   transform=transform, **kwargs)

    elif scalar:
        if lower == -np.inf:
            if upper == np.inf:
                transform = ScaleAwareIdentity(jitter_scale=jitter_scale)
                return pm.Flat(name, transform=transform, **kwargs)
            else:
                BoundedFlat = pm.Bound(pm.Flat, upper=upper)
                transform = ScaleAwareUpperBound(
                    upper,
                    testval=kwargs.get('testval', None),
                    jitter_scale=jitter_scale,
                )
                return BoundedFlat(name, transform=transform, **kwargs)
        else:
            assert upper == np.inf
            BoundedFlat = pm.Bound(pm.Flat, lower=lower)
            transform = ScaleAwareLowerBound(
                lower,
                testval=kwargs.get('testval', None),
                jitter_scale=jitter_scale,
            )
            return BoundedFlat(name, transform=transform, **kwargs)

    else:
        raise NotImplementedError('in the non-scalar case, unbounded supports '
                                  'have not been implemented yet.')

class ScaleAwareInterval(pm.distributions.transforms.Interval):
    name = "scaleawareinterval"
    def __init__(self, a, b, *,
                 jitter_scale=None, testval=None,
                 lerp_method='convex', scalar=None):
        super().__init__(a, b)

        if scalar is None:
            shape = _variable_shape(a, b, testval)
            scalar = shape == ()

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

        self.alpha = ScaleAwareInterval.compute_alpha(self.a, self.b,
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
        if self.alpha is None:
            return y
        else:
            return np.multiply(self.alpha, y, dtype=y.dtype)

    def backward(self, x):
        a, b = self.a, self.b
        if self.alpha is not None:
            x = x / self.alpha
        f = tt.nnet.sigmoid(x)
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


class ScaleAwareLowerBound(pm.distributions.transforms.LowerBound):
    # NB must derive from LowerBound otherwise cannot be pickled (reason unknown)
    #    must also call super() as much as possible otherwise cannot be pickled (reason unknown)
    name = "scaleawarelowerbound"

    def __init__(self, a, *, jitter_scale=None, testval=None):
        super().__init__(a)
        if np.asarray(a).ndim != 0:
            raise NotImplementedError('ScaleAwareLowerBound implemented only for float a')
        if jitter_scale is not None and testval is None:
            raise ValueError(
                'if testval is not given, jitter_scale cannot be given'
            )
        if jitter_scale is not None:
            if jitter_scale <= 0:
                raise ValueError('jitter_scale must be > 0')
            if testval <= a:
                raise ValueError('testval cannot be <= a')
            alpha = jitter_scale / (testval - a)
        else:
            alpha = 1.0
        self.alpha = floatX(alpha)

    def backward(self, x):
        return super().backward(self.alpha * x)

    def forward(self, x):
        return super().forward(x) / self.alpha

    def forward_val(self, x, point=None):
        return floatX(super().forward_val(x, point=point) / self.alpha)

    def jacobian_det(self, x):
        return super().jacobian_det(self.alpha * x) + tt.log(self.alpha)


class ScaleAwareUpperBound(pm.distributions.transforms.UpperBound):
    # NB must derive from UpperBound otherwise cannot be pickled (reason unknown)
    #    must also call super() as much as possible otherwise cannot be pickled (reason unknown)
    name = "scaleawareupperbound"

    def __init__(self, b, *, jitter_scale=None, testval=None):
        super().__init__(b)
        if np.asarray(b).ndim != 0:
            raise NotImplementedError('ScaleAwareUpperBound implemented only for float b')
        if jitter_scale is not None and testval is None:
            raise ValueError(
                'if testval is not given, jitter_scale cannot be given'
            )
        if jitter_scale is not None:
            if jitter_scale <= 0:
                raise ValueError('jitter_scale must be > 0')
            if testval >= b:
                raise ValueError('testval cannot be >= b')
            alpha = jitter_scale / (b - testval)
        else:
            alpha = 1.0
        self.alpha = floatX(alpha)

    def backward(self, x):
        return super().backward(self.alpha * x)

    def forward(self, x):
        return super().forward(x) / self.alpha

    def forward_val(self, x, point=None):
        return floatX(super().forward_val(x, point=point) / self.alpha)

    def jacobian_det(self, x):
        return super().jacobian_det(self.alpha * x) + tt.log(self.alpha)


class Identity(pm.distributions.transforms.ElemwiseTransform):
    """Identity transformation."""

    name = "identity"

    def backward(self, x):
        return x

    def forward(self, x):
        return x

    def forward_val(self, x, point=None):
        return x

    def jacobian_det(self, x):
        return floatX(0.0)


class ScaleAwareIdentity(pm.distributions.transforms.ElemwiseTransform):
    name = "scaleawareidentity"

    def __init__(self, *, jitter_scale=None):
        super().__init__()
        if jitter_scale is not None:
            if jitter_scale <= 0:
                raise ValueError('jitter_scale must be > 0')
            self.alpha = tt.as_tensor_variable(jitter_scale)
        else:
            self.alpha = floatX(1.0)

    def backward(self, x):
        return self.alpha * x

    def forward(self, x):
        return x / self.alpha

    def forward_val(self, x, point=None):
        return floatX(x / self.alpha)

    def jacobian_det(self, x):
        return tt.log(self.alpha)
