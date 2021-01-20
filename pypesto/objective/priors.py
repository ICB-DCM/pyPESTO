import numpy as np
from typing import Callable, Dict, List, Sequence, Tuple
from copy import deepcopy

from .function import ObjectiveBase
from .aggregated import AggregatedObjective
from .constants import MODE_FUN, MODE_RES, FVAL, GRAD, HESS

from .base import ResultDict


class NegLogPriors(AggregatedObjective):
    """
    Aggregates different forms of negative log-prior distributions.

    Allows to distinguish priors from the likelihood by testing the type of
    an objective.

    Consists basically of a list of individual negative log-priors,
    given in self.objectives.
    """


class NegLogParameterPriors(ObjectiveBase):
    """
    This class implements Negative Log Priors on Parameters.

    Contains a list of prior dictionaries for the individual parameters
    of the format

    {'index': [int],
    'density_fun': [Callable],
    'density_dx': [Callable],
    'density_ddx': [Callable]}

    A prior instance can be added to e.g. an objective, that gives the
    likelihood, by an AggregatedObjective.

    Notes
    -----

    All callables should correspond to log-densities. That is, they return
    log-densities and their corresponding derivatives.
    Internally, values are multiplied by -1, since pyPESTO expects the
    Objective function to be of a negative log-density type.
    """

    def __init__(self,
                 prior_list: List[Dict],
                 x_names: Sequence[str] = None):
        """
        Constructor

        Parameters
        ----------

        prior_list:
            List of dicts containing the individual parameter priors.
            Format see above.

        x_names:
            Sequence of parameter names (optional).
        """

        self.prior_list = prior_list
        super().__init__(x_names)

    def __deepcopy__(self, memodict=None):
        other = NegLogParameterPriors(deepcopy(self.prior_list))
        return other

    def call_unprocessed(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str
    ) -> ResultDict:

        res = {}

        res[FVAL] = self.neg_log_density(x)

        for order in sensi_orders:
            if order == 0:
                continue
            elif order == 1:
                res[GRAD] = self.gradient_neg_log_density(x)
            elif order == 2:
                res[HESS] = self.hessian_neg_log_density(x)
            else:
                raise ValueError(f'Invalid sensi order {order}.')

        return res

    def check_sensi_orders(self,
                           sensi_orders: Tuple[int, ...],
                           mode: str) -> bool:
        for order in sensi_orders:
            if not (0 <= order <= 2):
                return False
        return True

    def check_mode(self, mode) -> bool:
        if mode == MODE_FUN:
            return True
        elif mode == MODE_RES:
            return False
        else:
            raise ValueError(f'Invalid input: Expected mode {MODE_FUN} or'
                             f' {MODE_RES}, received {mode} instead.')

    def neg_log_density(self, x):
        """
        Computes the negative log-density for a parameter
        vector x.
        """
        density_val = 0
        for prior in self.prior_list:
            density_val -= prior['density_fun'](x[prior['index']])

        return density_val

    def gradient_neg_log_density(self, x):
        """
        Computes the gradient of the negative log-density for a parameter
        vector x.
        """
        grad = np.zeros_like(x)

        for prior in self.prior_list:
            grad[prior['index']] -= prior['density_dx'](x[prior['index']])

        return grad

    def hessian_neg_log_density(self, x):
        """
        Computes the hessian of the negative log-density for a parameter
        vector x.
        """
        hessian = np.zeros((len(x), len(x)))

        for prior in self.prior_list:
            hessian[prior['index'], prior['index']] -= \
                prior['density_ddx'](x[prior['index']])

        return hessian

    def hessian_vp_neg_log_density(self, x, p):
        """
        Computes the hessian vector product of the hessian of the
        negative log-density for a parameter vector x with a vector p.
        """
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

    prior_type:
        Prior is defined in LINEAR=untransformed parameter space,
        unless it starts with "parameterScale". prior_type
        can be any of {"uniform", "normal", "laplace", "logNormal",
        "parameterScaleUniform", "parameterScaleNormal",
        "parameterScaleLaplace"}

    prior_parameters:
        Parameters of the priors. Parameters are defined in linear scale.

    parameter_scale:
        scale in which the parameter is defined (since a parameter can be
        log-transformed, while the prior is always defined in the linear
        space, unless it starts with "parameterScale")
    """

    log_f, d_log_f_dx, dd_log_f_ddx = \
        _prior_densities(prior_type, prior_parameters)

    if parameter_scale == 'lin' or prior_type.startswith('parameterScale'):

        return {'index': index,
                'density_fun': log_f,
                'density_dx': d_log_f_dx,
                'density_ddx': dd_log_f_ddx}

    elif parameter_scale == 'log':

        def log_f_log(x_log):
            """log-prior for log-parameters"""
            return log_f(np.exp(x_log))

        def d_log_f_log(x_log):
            """derivative of log-prior w.r.t. log-parameters"""
            return d_log_f_dx(np.exp(x_log)) * np.exp(x_log)

        def dd_log_f_log(x_log):
            """second derivative of log-prior w.r.t. log-parameters"""
            return np.exp(x_log) * \
                (d_log_f_dx(np.exp(x_log)) +
                    np.exp(x_log) * dd_log_f_ddx(np.exp(x_log)))

        return {'index': index,
                'density_fun': log_f_log,
                'density_dx': d_log_f_log,
                'density_ddx': dd_log_f_log}

    elif parameter_scale == 'log10':

        log10 = np.log(10)

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
        raise ValueError(f"NegLogPriors in parameters in scale "
                         f"{parameter_scale} are currently not supported.")


def _prior_densities(prior_type: str,
                     prior_parameters: np.array) -> [Callable,
                                                     Callable,
                                                     Callable]:
    """
    Returns a tuple of Callables of the (log-)density (in untransformed =
    linear scale), unless prior_types starts with "parameterScale",
    together with their first + second derivative (= sensis) w.r.t.
    the parameters.

    Currently the following distributions are supported:

    Parameters
    ----------

    prior_type:
        string identifier indicating the distribution to be used. Here
        "transformed" parameter scale refers to the scale in which
        optimization is performed. For example, for parameters with scale
        "log", "parameterScaleNormal" will apply a normally distributed prior
        to logarithmic parameters, while "normal" will apply a normally
        distributed prior to linear parameters. For parameters with scale
        "lin", "parameterScaleNormal" and "norma" are equivalent.

        * uniform:
            Uniform distribution on transformed parameter scale.
        * parameterScaleUniform:
            Uniform distribution on original parameter scale.
        * normal:
            Normal distribution on transformed parameter scale.
        * parameterScaleNormal:
            Normal distribution on original parameter scale.
        * laplace:
            Laplace distribution on transformed parameter scale
        * parameterScaleLaplace:
            Laplace distribution on original parameter scale.
        * logNormal:
            LogNormal distribution on transformed parameter scale

        Currently not supported, but eventually planned are the
        following distributions:

        * logUniform
        * logLaplace

    prior_parameters:
        parameters for the distribution

        * uniform/parameterScaleUniform:
            - prior_parameters[0]: lower bound
            - prior_parameters[1]: upper bound

        * laplace/parameterScaleLaplace:
            - prior_parameters[0]: location parameter
            - prior_parameters[1]: scale parameter

        * normal/parameterScaleNormal:
            - prior_parameters[0]: mean
            - prior_parameters[1]: standard deviation

        * logNormal:
            - prior_parameters[0]: mean of log-parameters
            - prior_parameters[1]: standard deviation of log-parameters


    """

    if prior_type in ['uniform', 'parameterScaleUniform']:

        def log_f(x):
            if prior_parameters[0] <= x <= prior_parameters[1]:
                return - np.log(prior_parameters[1] - prior_parameters[0])
            else:
                return 0

        d_log_f_dx = _get_constant_function(0)
        dd_log_f_ddx = _get_constant_function(0)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type in ['normal', 'parameterScaleNormal']:

        mean = prior_parameters[0]
        sigma2 = prior_parameters[1]**2

        def log_f(x):
            return -np.log(2*np.pi*sigma2)/2 - \
                   (x-mean)**2/(2*sigma2)

        d_log_f_dx = _get_linear_function(-1/sigma2,
                                          mean/sigma2)
        dd_log_f_ddx = _get_constant_function(-1/sigma2)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type in ['laplace', 'parameterScaleLaplace']:

        mean = prior_parameters[0]
        scale = prior_parameters[1]
        log_2_sigma = np.log(2*prior_parameters[1])

        def log_f(x):
            return -log_2_sigma -\
                   abs(x-mean)/scale

        def d_log_f_dx(x):
            if x > prior_parameters[0]:
                return -1/scale
            else:
                return 1/scale

        dd_log_f_ddx = _get_constant_function(0)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'logUniform':
        # when implementing: add to tests
        raise NotImplementedError
    elif prior_type == 'logNormal':

        # TODO check again :)
        mean = prior_parameters[0]
        sigma = prior_parameters[1]
        sqrt2_pi = np.sqrt(2*np.pi)

        def log_f(x):
            return - np.log(sqrt2_pi * sigma * x) \
                   - (np.log(x) - mean)**2/(2*sigma**2)

        def d_log_f_dx(x):
            return - 1/x - (np.log(x) - mean)/(sigma**2 * x)

        def dd_log_f_ddx(x):
            return 1/(x**2) \
                   - (1 - np.log(x) + mean)/(sigma**2 * x**2)

        return log_f, d_log_f_dx, dd_log_f_ddx

    elif prior_type == 'logLaplace':
        # when implementing: add to tests
        raise NotImplementedError
    else:
        raise ValueError(f'NegLogPriors of type {prior_type} are currently '
                         'not supported')


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
    Defines a callable, that returns the constant, regardless of the input.
    """
    def function(x):
        return constant
    return function
