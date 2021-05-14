import numpy as np

from ..objective import Objective
from ..problem import Problem

try:
    import theano.tensor as tt
    try:
        from theano.graph.null_type import NullType
    except ImportError:
        # for older versions of theano
        from theano.gof.null_type import NullType
except ImportError:
    tt = NullType = None


class TheanoLogProbability(tt.Op):
    """
    Theano wrapper around a (non-normalized) log-probability function.

    Parameters
    ----------
    problem:
        The `pypesto.Problem` to analyze.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log prob)

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective

        # initialize the log probability Op
        self._log_prob = \
            lambda x: - beta * self._objective(x, sensi_orders=(0,))

        # initialize the sensitivity Op
        if problem.objective.has_grad:
            self._log_prob_grad = TheanoLogProbabilityGradient(problem, beta)
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


class TheanoLogProbabilityGradient(tt.Op):
    """
    Theano wrapper around a (non-normalized) log-probability gradient function.
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.

    Parameters
    ----------
    problem:
        The `pypesto.Problem` to analyze.
    beta:
        Inverse temperature (e.g. in parallel tempering).
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dvector]  # outputs a vector (the log prob grad)

    def __init__(self, problem: Problem, beta: float = 1.):
        self._objective: Objective = problem.objective
        self._log_prob_grad = \
            lambda x: - beta * self._objective(x, sensi_orders=(1,))

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate gradients
        log_prob_grad = self._log_prob_grad(theta)
        outputs[0][0] = log_prob_grad
