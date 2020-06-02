import numpy as np
from typing import Sequence, Callable
from copy import deepcopy
import math

from .function import ObjectiveBase
from .aggregated import AggregatedObjective


class Priors(AggregatedObjective):
    """
    Handles prior distributions.

    Consists basically of a list of individual priors,
    given in self.objectives.
    """


class ParameterPriors(ObjectiveBase):
    """
    Single Parameter Prior.


    prior_list has to contain dicts of the format format
    {'index': [int], 'density_fun': [Callable],
    'density_dx': [Callable], 'density_ddx': [Callable]}


    Note:
    -----
    All callables should correspond to (log)densities and are internally
    multiplied by -1, since pyPESTO performs minimization...
    """

    def __init__(self,
                 prior_list: list,
                 x_names: Sequence[str] = None):

        self.prior_list = prior_list

        super().__init__(fun=self.density_for_full_parameter_vector,
                         grad=self.gradient_for_full_parameter_vector,
                         hess=self.hessian_for_full_parameter_vector,
                         hessp=self.hessian_vp_for_full_parameter_vector)

    def __deepcopy__(self, memodict={}):
        other = ParameterPriors(deepcopy(self.prior_list))
        return other

    def density_for_full_parameter_vector(self, x):

        density_val = 0
        for prior in self.prior_list:
            density_val -= prior['density_fun'](x[prior['index']])

        return density_val

    def gradient_for_full_parameter_vector(self, x):

        grad = np.zeros_like(x)

        for prior in self.prior_list:
            grad[prior['index']] -= prior['density_dx'](x[prior['index']])

        return grad

    def hessian_for_full_parameter_vector(self, x):

        hessian = np.zeros((len(x), len(x)))

        for prior in self.prior_list:
            hessian[prior['index'], prior['index']] -= \
                prior['density_ddx'](x[prior['index']])

        return hessian

    def hessian_vp_for_full_parameter_vector(self, x, p):

        h_dot_p = np.zeros_like(p)

        for prior in self.prior_list:
            h_dot_p[prior['index']] -= \
                prior['density_ddx'](x[prior['index']]) * p[prior['index']]

        return h_dot_p


def get_parameter_prior_dict(index: int,
                             prior_type: str,
                             prior_parameters: list,
                             parameter_scale: str = 'lin'):

    """
    Returns the prior dict used to define priors for some default priors.

    index:
        index of the parameter in x_full

    prior_type: str
        Prior is defined in LINEAR parameter space! prior_type can from
        {uniform, normal, laplace, logUniform, logNormal, logLaplace}

    prior_parameters:
        Parameters of the priors. Parameters are defined in linear scale.

    parameter_scale:
        scale, in which parameter is defined (since a parameter can be
        log-transformed, while the prior is always defined in the linear space)
    """

    log_f, d_log_f_dx, dd_log_f_ddx = \
        _prior_densities(prior_type, prior_parameters)

    if parameter_scale == 'lin':

        return {'index': index,
                'density_fun': log_f,
                'density_dx': d_log_f_dx,
                'density_ddx': dd_log_f_ddx}

    elif parameter_scale == 'log':

        def log_f_log(x_log):
            """log-prior for log-parameters"""
            return log_f(math.exp(x_log))

        def d_log_f_log(x_log):
            """derivative of log-prior w.r.t. log-parameters"""
            return d_log_f_dx(math.exp(x_log)) * math.exp(x_log)

        def dd_log_f_log(x_log):
            """second derivative of log-prior w.r.t. log-parameters"""
            return math.exp(x_log) * \
                (d_log_f_dx(math.exp(x_log)) +
                    math.exp(x_log) * dd_log_f_ddx(math.exp(x_log)))

        return {'index': index,
                'density_fun': log_f_log,
                'density_dx': d_log_f_log,
                'density_ddx': dd_log_f_log}

    elif parameter_scale == 'log10':

        log10 = math.log(10)

        def log_f_log10(x_log10):
            """log-prior for log10-parameters"""
            return log_f(10**x_log10)

        def d_log_f_log10(x_log10):
            """derivative of log-prior w.r.t. log10-parameters"""
            return d_log_f_dx(10**x_log10) * log10 * 10**x_log10

        def dd_log_f_log10(x_log10):
            """second derivative of log-prior w.r.t. log10-parameters"""
            return log10**2 * 10**x_log10 * \
                (dd_log_f_ddx(10**x_log10) * 10**x_log10
                    + d_log_f_dx(10**x_log10))

        return {'index': index,
                'density_fun': log_f_log10,
                'density_dx': d_log_f_log10,
                'density_ddx': dd_log_f_log10}

    else:
        raise ValueError(f"Priors in parameters in scale {parameter_scale}"
                         f" are currently not supported.")


def _prior_densities(prior_type: str,
                     prior_parameters: np.array) -> [Callable,
                                                     Callable,
                                                     Callable]:
    """
    Returns a tuple of Callables of the (log-)density (in linear scale),
    together with their first + second derivative (= senisis) w.r.t x
    """

    if prior_type == 'uniform':

        log_f = _get_constant_function(
            - math.log(prior_parameters[1] - prior_parameters[0]))
        d_log_f_dx = _get_constant_function(0)
        dd_log_f_ddx = _get_constant_function(0)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'normal':

        sigma2 = prior_parameters[1]**2

        def log_f(x):
            return -math.log(2*math.pi*sigma2)/2 - \
                   (x-prior_parameters[0])**2/(2*sigma2)

        d_log_f_dx = _get_linear_function(-1/sigma2,
                                          prior_parameters[0]/sigma2)
        dd_log_f_ddx = _get_constant_function(-1/sigma2)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'laplace':
        log_2_sigma = math.log(2*prior_parameters[1])

        def log_f(x):
            return -log_2_sigma -\
                   abs(x-prior_parameters[0])/prior_parameters[1]

        def d_log_f_dx(x):
            if x > prior_parameters[0]:
                return -1/prior_parameters[1]
            else:
                return 1/prior_parameters[1]

        dd_log_f_ddx = _get_constant_function(0)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'logUniform':
        # when implementing: add to tests
        raise NotImplementedError
    elif prior_type == 'logNormal':

        sigma2 = prior_parameters[1]**2
        sqrt2_pi = math.sqrt(2*math.pi)

        def log_f(x):
            return - math.log(sqrt2_pi * prior_parameters[1] * x) \
                   - (math.log(x) - prior_parameters[0])**2/(2*sigma2)

        def d_log_f_dx(x):
            return - 1/x - (math.log(x) - prior_parameters[0])/(sigma2 * x)

        def dd_log_f_ddx(x):
            return 1/(x**2) \
                   - (1 - math.log(x) + prior_parameters[0])/(sigma2 * x**2)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'logLaplace':
        # when implementing: add to tests
        raise NotImplementedError
    else:
        ValueError(f'Priors of type {prior_type} are currently not supported')


def _get_linear_function(slope: float,
                         intercept: float = 0):
    """
    Returns a linear function
    """
    def function(x):
        return slope * x + intercept
    return function


def _get_constant_function(constant: float):
    """
    Defines a callable, that returns a callable, that returns
    the constant, regardless of the input
    """
    def function(x):
        return constant
    return function
