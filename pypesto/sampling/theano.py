from typing import Union

import numpy as np

from ..objective import ObjectiveBase
from ..problem import Problem

import theano.tensor as tt
from theano.gof.null_type import NullType


class CachedObjective:
    """
    Wrapper around an ObjectiveBase which computes the gradient at each evaluation,
    caching it for later calls.
    Caching is only enabled after the first time the gradient is asked for
    and disabled whenever the cached gradient is not used,
    in order not to increase computation time for derivative-free samplers.

    Parameters
    ----------
    objective:
        The `pypesto.ObjectiveBase` to wrap.
    """

    def __init__(self, objective: ObjectiveBase):
        self.objective = objective
        self.x_cached = None
        self.fval_cached = None
        self.grad_cached = None
        self.grad_has_been_used = False

    def __call__(self, x, sensi_orders):
        if sensi_orders == (0,) and self.x_cached is None:
            # The gradient has not been called yet: caching is off
            return self.objective(x, sensi_orders=sensi_orders)
        else:
            # Check if we hit the cache
            if not np.array_equal(x, self.x_cached):
                # If the currently cached gradient has never been used,
                # turn off caching
                if sensi_orders == (0,) and not self.grad_has_been_used:
                    self.x_cached = None
                    return self.objective(x, sensi_orders=sensi_orders)
                # Repopulate cache
                fval, grad = self.objective(x, sensi_orders=(0, 1))
                self.x_cached = x  # NB it seems that at each call x is
                                   # a different object, so it is safe
                                   # not to copy it
                self.fval_cached = fval
                self.grad_cached = grad
                self.grad_has_been_used = False
            # The required values are in the cache
            if sensi_orders == (0,):
                return self.fval_cached
            elif sensi_orders == (1,):
                self.grad_has_been_used = True
                return self.grad_cached
            elif sensi_orders == (0, 1):
                self.grad_has_been_used = True
                return self.fval_cached, self.grad_cached
            else:
                raise NotImplementedError(f'sensi_orders = {sensi_orders}')

    @property
    def has_grad(self):
        return self.objective.has_grad


class TheanoLogProbability(tt.Op):
    """
    Theano wrapper around a (non-normalized) log-probability function.

    Parameters
    ----------
    problem:
        The `pypesto.ObjectiveBase` defining the log-probability.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log prob)

    def __init__(self, objective: Union[ObjectiveBase, CachedObjective], beta: float = 1.):
        self._objective = objective
        self._beta = beta

        # initialize the sensitivity Op
        if objective.has_grad:
            self._log_prob_grad = TheanoLogProbabilityGradient(objective, beta)
        else:
            self._log_prob_grad = None

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        log_prob = -self._beta * self._objective(theta, sensi_orders=(0,))
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


class TheanoLogProbabilityGradient(tt.Op):
    """
    Theano wrapper around a (non-normalized) log-probability gradient function.
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.

    Parameters
    ----------
    problem:
        The `pypesto.ObjectiveBase` defining the log-probability.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dvector]  # outputs a vector (the log prob grad)

    def __init__(self, objective: Union[ObjectiveBase, CachedObjective], beta: float = 1.):
        self._objective = objective
        self._beta = beta

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate gradients
        log_prob_grad = -self._beta * self._objective(theta, sensi_orders=(1,))
        outputs[0][0] = log_prob_grad
