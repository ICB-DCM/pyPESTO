"""
Utilities for creating a pymc3.Model from a pypesto.Problem.
"""

import math
from typing import Tuple, List, Optional

import numpy as np

import pymc3 as pm
import theano.tensor as tt

from ...problem import Problem
from .theano import TheanoLogProbability, CachedObjective
from .interval import ScaleAwareUniform, Identity

PYMC3_LOGPOST = 'log_post'
PYMC3_THETA = 'theta'


def create_pymc3_model(problem: Problem,
                       testval: Optional[np.ndarray] = None,
                       beta: float = 1., *,
                       support: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                       jitter_scales: Optional[np.ndarray] = None,
                       cache_gradients: bool = True,
                       vectorize: bool = True,
                       remap_to_reals: bool = True,
                       lerp_method: str = 'convex',
                       check: bool = True,
                       verbose: bool = False):
        # Check consistency of arguments
        if jitter_scales is not None:
            if testval is None:
                raise ValueError('if testval is not given, '
                                 'jitter_scales cannot be given')
            if not remap_to_reals:
                raise ValueError('if remap_to_reals is False, '
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
        else:
            lbs, ubs = problem.lb, problem.ub

        # Convert to numpy arrays and check sizes
        lbs, ubs = np.asarray(lbs), np.asarray(ubs)
        if testval is not None:
            testval = np.asarray(testval)
        if jitter_scales is not None:
            jitter_scales = np.asarray(jitter_scales)
        if len(lbs) != problem.dim or len(ubs) != problem.dim:
            raise ValueError('length of lower/upper bounds '
                             'must be equal to number of free parameters')
        if testval is not None and len(testval) != problem.dim:
            raise ValueError('The size of the test value must be equal ' \
                             'to the number of free parameters')
        if jitter_scales is not None and len(jitter_scales) != problem.dim:
            raise ValueError('The size of jitter_scales must be equal ' \
                             'to the number of free parameters')

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
                        raise ValueError(f'testval[{i}] (parameter: {problem.x_names[i]}) is out of bounds ({lb}, {ub})')
                    else:
                        # Move this parameter inside the interval
                        # by taking the nearest floating point value
                        # (it appears this is enough to solve the problem)
                        if x == lb:
                            testval[i] = np.nextafter(lb, ub)
                        else:
                            assert x == ub
                            testval[i] = np.nextafter(ub, lb)

            # Create a uniform bounded vector variable for all parameters
            if vectorize:
                if remap_to_reals:
                    if testval is None:
                        theta = ScaleAwareUniform(pymc3_vector_parname(x_names),
                                              lower=lbs, upper=ubs,
                                              lerp_method=lerp_method)
                    else:
                        theta = ScaleAwareUniform(pymc3_vector_parname(x_names),
                                              lower=lbs, upper=ubs,
                                              testval=testval,
                                              jitter_scale=jitter_scales,
                                              lerp_method=lerp_method)
                elif testval is None:
                    theta = pm.Uniform(pymc3_vector_parname(x_names),
                                       lower=lbs, upper=ubs,
                                       shape=(problem.dim,),
                                       transform=Identity())
                else:
                    theta = pm.Uniform(pymc3_vector_parname(x_names),
                                       lower=lbs, upper=ubs,
                                       shape=(problem.dim,),
                                       transform=Identity(),
                                       testval=testval)

            # Create a uniform bounded variable for each parameter
            elif remap_to_reals:
                if testval is None:
                    k = [ScaleAwareUniform(x_name, lower=lb, upper=ub,
                                           lerp_method=lerp_method)
                             for x_name, lb, ub in
                             zip(x_names, lbs, ubs)]
                elif jitter_scales is None:
                    k = [ScaleAwareUniform(x_name, lower=lb, upper=ub,
                                           testval=x, lerp_method=lerp_method)
                             for x_name, x, lb, ub in
                             zip(x_names, testval, lbs, ubs)]
                else:
                    k = [ScaleAwareUniform(x_name, lower=lb, upper=ub,
                                           testval=x, jitter_scale=jitter_scale,
                                           lerp_method=lerp_method)
                             for x_name, x, lb, ub, jitter_scale in
                             zip(x_names, testval, lbs, ubs, jitter_scales)]
            elif testval is None:
                k = [pm.Uniform(x_name, lower=lb, upper=ub, transform=None,
                                testval=_reasonable_testval(lb, ub))
                         for x_name, lb, ub in
                         zip(x_names, lbs, ubs)]
            else:
                k = [pm.Uniform(x_name, lower=lb, upper=ub, transform=None,
                                testval=x)
                         for x_name, x, lb, ub in
                         zip(x_names, testval, lbs, ubs)]


            # Convert to tensor vector
            if not vectorize:
                theta = tt.as_tensor_variable(k)

            # Use a DensityDist for the log-posterior
            # TODO PyMC3 does not allow a free variable to be passed as the observed value,
            #      but tensor variable or TransformedRV are accepted (the latter case is maybe a bug?)
            #      So this is really unsupported and potentially dangerous behaviour...
            #      Maybe pymc3.Potential can be a cleaner solution?
            pm.DensityDist(pymc3_logp_parname(x_names),
                           logp=log_post_fun, observed=theta)

        # Check posterior at testval
        if check:
            logps = [RV.logp(model.test_point) for RV in model.basic_RVs]
            if not all(math.isfinite(logp) for logp in logps):
                raise Exception('Log-posterior of same basic random variables' \
                                ' is not finite. Please report this issue at ' \
                                'https://github.com/ICB-DCM/pyPESTO/issues' \
                                '\nLog-posterior at test point is\n' + \
                                str(model.check_test_point()))

        if verbose:
            print('Evaluating log-posterior at test point')
            print(model.check_test_point())

        return model


def _reasonable_testval(lb, ub):
    return np.where(np.isinf(lb), np.where(np.isinf(ub), 0.0, ub - 1.0), np.where(np.isinf(ub), lb + 1.0, (lb + ub) / 2))


def pymc3_vector_parname(x_names: List[str]):
    if PYMC3_THETA in x_names:
        if 'pymc3_' + PYMC3_THETA in x_names:
            raise Exception('cannot find a name for the parameter vector')
        else:
            return 'pymc3_' + PYMC3_THETA
    else:
        return PYMC3_THETA


def pymc3_logp_parname(x_names: List[str]):
    if PYMC3_LOGPOST in x_names:
        if 'pymc3_' + PYMC3_LOGPOST in x_names:
            raise Exception('cannot find a name for the log-posterior')
        else:
            return 'pymc3_' + PYMC3_LOGPOST
    else:
        return PYMC3_LOGPOST

def pypesto_varnames(model: pm.Model, problem: Problem):
    x_names = [problem.x_names[i] for i in problem.x_free_indices]
    # Determine if free parameters were vectorized
    if len(model.free_RVs) == 1:
        theta = model.free_RVs[0]
        if len(theta.distribution.shape) > 0:
            # theta is a tensor variable
            assert len(theta.distribution.shape) == 1  # should only be vector
            return [pymc3_vector_parname(x_names)]
    return x_names
