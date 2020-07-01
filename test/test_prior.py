
"""
This is for testing priors.
"""

import math
import itertools
import numpy as np
import scipy.optimize as opt

import pypesto
import pypesto.optimize
from pypesto.objective import NegLogParameterPriors
from pypesto.objective.priors import get_parameter_prior_dict


def test_mode():
    """
    Tests the maximum/optimum for priors in different scales...
    """

    scales = ['lin', 'log', 'log10']
    prior_types = ['normal', 'laplace',
                   'logNormal']

    problem_dict = {'lin': {'lb': [0], 'ub': [10], 'opt': [1]},
                    'log': {'lb': [-3], 'ub': [3], 'opt': [0]},
                    'log10': {'lb': [-3], 'ub': [2], 'opt': [0]}}

    for prior_type, scale in itertools.product(prior_types, scales):

        prior_list = [get_parameter_prior_dict(
            0, prior_type, [1, 1], scale)]

        test_prior = NegLogParameterPriors(prior_list)
        test_problem = pypesto.Problem(test_prior,
                                       lb=problem_dict[scale]['lb'],
                                       ub=problem_dict[scale]['ub'],
                                       dim_full=1,
                                       x_scales=[scale])

        optimizer = pypesto.optimize.ScipyOptimizer(method='Nelder-Mead')

        result = pypesto.optimize.minimize(
            problem=test_problem, optimizer=optimizer, n_starts=10)

        assert np.isclose(result.optimize_result.list[0]['x'],
                          problem_dict[scale]['opt'], atol=1e-04)

    # test uniform distribution:
    for scale in scales:
        prior_dict = get_parameter_prior_dict(
            0, 'uniform', [1, 2], scale)

        # check inside and outside of interval
        assert abs(prior_dict['density_fun'](lin_to_scaled(.5, scale))
                   - 0) < 1e-8

        assert abs(prior_dict['density_fun'](lin_to_scaled(1.5, scale))
                   - math.log(1)) < 1e-8

        assert abs(prior_dict['density_fun'](lin_to_scaled(2.5, scale))
                   - 0) < 1e-8


def test_derivatives():
    """
    Tests the finite gradients and second order derivatives.
    """

    scales = ['lin', 'log', 'log10']
    prior_types = ['uniform', 'normal', 'laplace', 'logNormal']

    for prior_type, scale in itertools.product(prior_types, scales):

        if prior_type == 'uniform':
            prior_parameters = [-1, 1]
        else:
            prior_parameters = [1, 1]

        prior_dict = get_parameter_prior_dict(
            0, prior_type, prior_parameters, scale)

        # use this x0, since it is a moderate value both in linear
        # and in log scale...
        x0 = np.array([0.5])

        err_grad = opt.check_grad(prior_dict['density_fun'],
                                  prior_dict['density_dx'], x0)
        err_hes = opt.check_grad(prior_dict['density_dx'],
                                 prior_dict['density_ddx'], x0)

        assert err_grad < 1e-3
        assert err_hes < 1e-3


def lin_to_scaled(x: float,
                  scale: str):
    """
    transforms x to linear scale
    """
    if scale == 'lin':
        return x
    elif scale == 'log':
        return math.log(x)
    elif scale == 'log10':
        return math.log10(x)
    else:
        ValueError(f'Unknown scale {scale}')


def scaled_to_lin(x: float,
                  scale: str):
    """
    transforms x to scale
    """
    if scale == 'lin':
        return x
    elif scale == 'log':
        return math.exp(x)
    elif scale == 'log10':
        return 10**x
    else:
        ValueError(f'Unknown scale {scale}')
