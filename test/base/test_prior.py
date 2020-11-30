"""Test priors."""

import math
import itertools
import pytest
import numpy as np
import scipy.optimize as opt

import pypesto
import pypesto.optimize
from pypesto.objective import NegLogParameterPriors
from pypesto.objective.priors import get_parameter_prior_dict

scales = ['lin', 'log', 'log10']


@pytest.fixture(params=scales)
def scale(request):
    return request.param


prior_types = ['uniform', 'normal', 'laplace', 'logNormal',
               'parameterScaleUniform', 'parameterScaleNormal',
               'parameterScaleLaplace']


@pytest.fixture(params=prior_types)
def prior_type(request):
    return request.param


def test_mode(scale, prior_type):
    """
    Tests the maximum/optimum for priors in different scales...
    """

    problem_dict = {'lin': {'lb': [0], 'ub': [10], 'opt': [1]},
                    'log': {'lb': [-3], 'ub': [3], 'opt': [0]},
                    'log10': {'lb': [-3], 'ub': [2], 'opt': [0]}}

    prior_list = [get_parameter_prior_dict(
        0, prior_type, [1, 1], scale)]

    test_prior = NegLogParameterPriors(prior_list)
    test_problem = pypesto.Problem(test_prior,
                                   lb=problem_dict[scale]['lb'],
                                   ub=problem_dict[scale]['ub'],
                                   dim_full=1,
                                   x_scales=[scale])

    if prior_type.startswith('parameterScale'):
        scale = 'lin'

    optimizer = pypesto.optimize.ScipyOptimizer(method='Nelder-Mead')

    result = pypesto.optimize.minimize(
        problem=test_problem, optimizer=optimizer, n_starts=10)

    # test uniform distribution:
    if prior_type in ['uniform', 'parameterScaleUniform']:

        if prior_type == 'parameterScaleUniform':
            scale = 'lin'

        # check inside and outside of interval
        assert abs(prior_list[0]['density_fun'](
            lin_to_scaled(.5, scale)) - 0
        ) < 1e-8

        assert abs(prior_list[0]['density_fun'](
            lin_to_scaled(1.5, scale)) - math.log(1)
        ) < 1e-8

        assert abs(prior_list[0]['density_fun'](
            lin_to_scaled(2.5, scale)) - 0
        ) < 1e-8

    else:
        # flat functions don't have local minima, so dont check this for
        # uniform priors
        assert np.isclose(result.optimize_result.list[0]['x'],
                          problem_dict[scale]['opt'], atol=1e-04)


def test_derivatives(prior_type, scale):
    """
    Tests the finite gradients and second order derivatives.
    """

    if prior_type in ['uniform', 'parameterScaleUniform']:
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
