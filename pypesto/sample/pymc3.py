import pymc3 as pm
import theano.tensor as tt
import numpy as np
from typing import Any, Dict, Union

from ..problem import Problem
from .sampler import Sampler
from .result import McmcPtResult


class Pymc3Sampler(Sampler):
    """Use Pymc3 for sampling."""

    def __init__(self, options: Dict = None, return_native: bool = False):
        """Constructor.

        Parameters
        ----------
        options:
            Options configuring the sampler run.
        return_native:
            Whether to return the result object in native Pymc3 format.
        """
        if options is None:
            options = {}
        self._options = options
        self._return_native = return_native

    def sample(self, problem: Problem) -> Union[McmcPtResult, Any]:
        # create our Op
        llh = TheanoLogLikelihood(problem)

        # use PyMC3 to sampler from log-likelihood
        with pm.Model():
            # uniform priors on m and c
            k = [pm.Uniform(x_name, lower=lb, upper=ub)
                 for x_name, lb, ub in
                 zip(problem.x_names, problem.lb, problem.ub)]

            # convert m and c to a tensor vector
            theta = tt.as_tensor_variable(k)

            # use a DensityDist (use a lamdba function to "call" the Op)
            pm.DensityDist('likelihood', lambda v: llh(v),
                           observed={'v': theta})

            trace = pm.sample(**self._options)

        if self._return_native:
            return trace

        # TODO
        raise NotImplementedError(
            "Conversion from PyMC3 to pyPESTO result format is not "
            "implemented yet.")


class TheanoLogLikelihood(tt.Op):
    """
    Theano wrapper around the log-likelihood function.
    """
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, problem: Problem):
        self._objective = problem.objective

        # initialize the llh Op
        self._llh = lambda x: - self._objective(x, sensi_orders=(0,))

        # initialize the sllh Op
        self._sllh = TheanoLogLikelihoodGradient(problem)

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        llh = self._llh(theta)
        outputs[0][0] = np.array(llh)

    def grad(self, inputs, g):
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        theta, = inputs
        sllh = self._sllh(theta)
        return [g[0] * sllh]


class TheanoLogLikelihoodGradient(tt.Op):
    """
    Theano wrapper around the log-likelihood gradient function.

    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """
    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, problem):
        self._objective = problem.objective
        self._sllh = lambda x: - self._objective(x, sensi_orders=(1,))

    def perform(self, node, inputs, outputs, params=None):
        theta, = inputs
        # calculate gradients
        sllh = self._sllh(theta)
        outputs[0][0] = sllh
