from typing import Union, Tuple, List
import numpy as np

import pymc3 as pm
import arviz as az
import theano.tensor as tt
from theano.ifelse import ifelse

from ..problem import Problem
from .theano import TheanoLogProbability, CachedObjective


def create_pymc3_model(problem: Problem,
                       testval: Union[np.ndarray, None] = None,
                       beta: float = 1., *,
                       support: Union[Tuple[np.ndarray, np.ndarray], None]=None,
                       cache_gradients: bool = True,
                       vectorize: bool = True,
                       lerp: str = 'convex',
                       verbose: bool = False):
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
                                          lerp=lerp, lower=lbs, upper=ubs)
                else:
                    theta = BetterUniform(pymc3_vector_parname(x_names),
                                          lerp=lerp, testval=testval,
                                          lower=lbs, upper=ubs)
            elif testval is None:
                k = [BetterUniform(x_name, lower=lb, upper=ub, lerp=lerp)
                         for x_name, lb, ub in
                         zip(x_names, lbs, ubs)]
            else:
                k = [BetterUniform(x_name, testval=x, lower=lb, upper=ub, lerp=lerp)
                         for x_name, x, lb, ub in
                         zip(x_names, testval, lbs, ubs)]

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


def pymc3_to_arviz(model: pm.Model, trace: pm.backends.base.MultiTrace,
                   *, problem: Problem = None, x_names: List[str] = None,
                   vectorize: bool = False):
    if problem is not None and x_names is not None:
        raise ValueError('the problem and x_names keyword arguments '
                         'cannot both be given.')

    kwargs = {}
    if vectorize:
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

    # Check bounds ordering
    if not (lower <= upper).all():
        raise ValueError('each lower bound should be smaller than '
                         'the corresponding upper bound')

    # Depending on the bounds, choose the transformation
    if np.isfinite(lower).all() and np.isfinite(upper).all():
        BoundedFlat = pm.Bound(pm.Flat, lower=lower, upper=upper)
        transform = BetterInterval(lower, upper, lerp, scalar=scalar)
        return BoundedFlat(name, transform=transform, **kwargs)
        # In the case we start from pm.Uniform,
        # we need to comment the jacobian out of BetterInterval
        # transform = BetterInterval(lower, upper, lerp, scalar=scalar)
        # return pm.Uniform(name, lower=lower, upper=upper, transform=transform, **kwargs)
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


class BetterInterval(pm.distributions.transforms.Interval):
    name = "betterinterval"
    def __init__(self, a, b, lerp, *, scalar=True):
        super().__init__(a, b)

        scalar_bounds = np.shape(a) == () and np.shape(b) == ()
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
