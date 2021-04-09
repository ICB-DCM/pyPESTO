"""
This adds an  interface for the construction of loss functions incorporating
aesara models
"""

import numpy as np

from . import Objective
from ..problem import Problem

try:
    import aesara.tensor as aet
    from aesara.graph.op import Op
    from aesara.graph.null_type import NullType
except ImportError:
    aet = NullType = Op = None


class AesaraLogProbability(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability function.

    Parameters
    ----------
    problem:
        The `pypesto.Problem` to analyze.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [aet.dvector]  # expects a vector of parameter values when called
    otypes = [aet.dscalar]  # outputs a single scalar value (the log prob)

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective

        def log_prob(x):
            return - beta * self._objective(x, sensi_orders=(0,))

        # initialize the log probability Op
        self._log_prob = log_prob

        # initialize the sensitivity Op
        if problem.objective.has_grad:
            self._log_prob_grad = AesaraLogProbabilityGradient(problem, beta)
        else:
            self._log_prob_grad = None

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        log_prob = self._log_prob(theta)
        outputs[0][0] = np.array(log_prob)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        if self._log_prob_grad is None:
            # indicates gradient not available
            return [NullType]
        theta, = inputs
        log_prob_grad = self._log_prob_grad(theta)
        return [g[0] * log_prob_grad]


class AesaraLogProbabilityGradient(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability gradient function.
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.

    Parameters
    ----------
    problem:
        The `pypesto.Problem` to analyze.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [aet.dvector]  # expects a vector of parameter values when called
    otypes = [aet.dvector]  # outputs a vector (the log prob grad)

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective
        self._nx = problem.dim_full

        def log_prob_grad(x):
            return - beta * self._objective(x, sensi_orders=(1,))

        self._log_prob_grad = log_prob_grad

        if problem.objective.has_hess:
            self._log_prob_hess = AesaraLogProbabilityHessian(problem, beta)
        else:
            self._log_prob_hess = None

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate gradients
        log_prob_grad = self._log_prob_grad(theta)
        outputs[0][0] = log_prob_grad

    def grad(self, inputs, g):
        # the method that calculates the hessian - it actually returns the
        # vector-hessian product - g[0] is a vector of parameter values
        if self._log_prob_hess is None:
            # indicates gradient not available
            return [NullType]
        theta, = inputs
        log_prob_hess = self._log_prob_hess(theta)
        return [g[0].dot(log_prob_hess)]


class AesaraLogProbabilityHessian(Op):
    """
    Aesara wrapper around a (non-normalized) log-probability Hessian function.
    This Op will be called with a vector of values and also return a matrix of
    values - the Hessian in each dimension.

    Parameters
    ----------
    problem:
        The `pypesto.Problem` to analyze.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [aet.dvector]
    otypes = [aet.dmatrix]

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective

        def _log_prob_hess(x):
            return - beta * self._objective(x, sensi_orders=(2,))

        self._log_prob_hess = _log_prob_hess

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate Hessian
        log_prob_hess = self._log_prob_hess(theta)
        outputs[0][0] = log_prob_hess
