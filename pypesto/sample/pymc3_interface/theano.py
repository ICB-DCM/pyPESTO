"""
Theano wrapper for pyPESTO objectives.
"""

from typing import Union, Tuple

import copy
import numpy as np

from pypesto.objective.base import ObjectiveBase, ResultDict
from pypesto.objective.constants import MODE_FUN, FVAL, GRAD
from ...problem import Problem

import theano.tensor as tt
from theano.gof.null_type import NullType


class CachedObjective(ObjectiveBase):
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
        if not isinstance(objective, ObjectiveBase):
            raise TypeError(f'objective must be an ObjectiveBase instance')
        if not objective.check_mode(MODE_FUN):
            raise NotImplementedError(f'objective must support mode={MODE_FUN}')
        super().__init__(objective.x_names)
        self.pre_post_processor = objective.pre_post_processor
        self.objective = objective
        self.x_cached = None
        self.fval_cached = None
        self.grad_cached = None
        self.grad_has_been_used = False

    def __deepcopy__(self, memodict=None) -> 'CachedObjective':
        return CachedObjective(copy.deepcopy(self.objective))

    def initialize(self):
        self.objective.initialize()
        self.x_cached = None
        self.fval_cached = None
        self.grad_cached = None
        self.grad_has_been_used = False

    def call_unprocessed(
            self,
            x: np.ndarray,
            sensi_orders: Tuple[int, ...],
            mode: str
        ) -> ResultDict:

        if sensi_orders == (0,) and self.x_cached is None:
            # The gradient has not been called yet: caching is off
            return self.objective.call_unprocessed(x, sensi_orders, mode)

        else:
            # Check if we hit the cache
            if not np.array_equal(x, self.x_cached):
                # If the currently cached gradient has never been used,
                # turn off caching
                if sensi_orders == (0,) and not self.grad_has_been_used:
                    self.x_cached = None
                    return self.objective.call_unprocessed(x, sensi_orders, mode)
                # Repopulate cache
                retval = self.objective.call_unprocessed(x, (0, 1), mode)
                self.x_cached = x  # NB it seems that at each call x is
                                   # a different object, so it is safe
                                   # not to copy it
                self.fval_cached = retval[FVAL]
                self.grad_cached = retval[GRAD]
                self.grad_has_been_used = False

            # The required values are in the cache
            if sensi_orders == (0,):
                return {FVAL: self.fval_cached}
            elif sensi_orders == (1,):
                self.grad_has_been_used = True
                return {GRAD: self.grad_cached}
            else:
                assert sensi_orders == (0, 1)  # this should be ensured by check_sensi_orders
                self.grad_has_been_used = True
                return {FVAL: self.fval_cached, GRAD: self.grad_cached}

    def check_mode(self, mode) -> bool:
        return mode == MODE_FUN

    def check_sensi_orders(self, sensi_orders, mode) -> bool:
        if max(sensi_orders) > 1 or mode != MODE_FUN:
            return False
        else:
            return self.objective.check_sensi_orders(sensi_orders, mode)


class TheanoLogProbability(tt.Op):
    """
    Theano wrapper around a (non-normalized) log-probability function.

    Parameters
    ----------
    problem:
        The `pypesto.ObjectiveBase` defining the log-probability.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    check_finite:
        Check parameter vector for non-finite elements;
        if any are found, set log-probability to -inf and gradient to NaN.
        May fix things in some situations, may break things in others.
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log prob)

    def __init__(self, objective: ObjectiveBase, beta: float = 1., check_finite: bool = True):
        self._objective = objective
        self._beta = beta
        self._check_finite = check_finite

        # initialize the sensitivity Op
        if objective.has_grad:
            self._log_prob_grad = TheanoLogProbabilityGradient(objective, beta, check_finite=check_finite)
        else:
            self._log_prob_grad = None

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        if not self._check_finite or np.all(np.isfinite(theta)):
            log_prob = -self._beta * self._objective(theta, sensi_orders=(0,))
        else:
            print('Parameter vector theta contains non-finite elements! Log-probability set to -inf.', file=sys.stderr)
            log_prob = -np.inf
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
    check_finite:
        Check parameter vector for non-finite elements;
        if any are found, set gradient to NaN.
        May fix things in some situations, may break things in others.
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dvector]  # outputs a vector (the log prob grad)

    def __init__(self, objective: ObjectiveBase, beta: float = 1., check_finite: bool = True):
        self._objective = objective
        self._beta = beta
        self._check_finite = check_finite

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        if not self._check_finite or np.all(np.isfinite(theta)):
            log_prob_grad = -self._beta * self._objective(theta, sensi_orders=(1,))
        else:
            print('Parameter vector theta contains non-finite elements! Log-probability gradient set to NaN.', file=sys.stderr)
            log_prob_grad = np.full(theta.shape, np.nan)
        outputs[0][0] = log_prob_grad
