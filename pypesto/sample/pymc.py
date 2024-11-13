"""Pymc v4 Sampler."""

from __future__ import annotations

import logging

import arviz as az
import numpy as np
import pymc
import pytensor.tensor as pt

from ..history import MemoryHistory
from ..objective import ObjectiveBase
from ..problem import Problem
from ..result import McmcPtResult
from .sampler import Sampler, SamplerImportError

logger = logging.getLogger(__name__)

# implementation based on:
# https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html


class PymcObjectiveOp(pt.Op):
    """PyTensor wrapper around a (non-normalized) log-probability function."""

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log prob)

    def create_instance(objective: ObjectiveBase, beta: float = 1.0):
        """Create an instance of this Op (factory method).

        Parameters
        ----------
        objective:
            Objective function (negative log-likelihood or -posterior).
        beta:
            Inverse temperature (e.g. in parallel tempering).

        Returns
        -------
        PymcObjectiveOp
            The created instance.
        """
        if objective.has_grad:
            return PymcObjectiveWithGradientOp(objective, beta)
        return PymcObjectiveOp(objective, beta)

    def __init__(self, objective: ObjectiveBase, beta: float = 1.0):
        self._objective: ObjectiveBase = objective
        self._beta: float = beta

    def perform(self, node, inputs, outputs, params=None):
        """Calculate the objective function value."""
        (theta,) = inputs
        log_prob = -self._beta * self._objective(theta, sensi_orders=(0,))
        outputs[0][0] = np.array(log_prob)


class PymcObjectiveWithGradientOp(PymcObjectiveOp):
    """PyTensor objective wrapper with gradient."""

    def __init__(self, objective: ObjectiveBase, beta: float = 1.0):
        super().__init__(objective, beta)

        self._log_prob_grad = PymcGradientOp(objective, beta)

    def grad(self, inputs, g):  # noqa
        """Calculate the vector-Jacobian product."""
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self._log_prob_grad(theta)]


class PymcGradientOp(pt.Op):
    """PyTensor wrapper around a (non-normalized) log-probability gradient."""

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dvector]  # outputs a vector (the log prob grad)

    def __init__(self, objective: ObjectiveBase, beta: float):
        self._objective: ObjectiveBase = objective
        self._beta: float = beta

    def perform(self, node, inputs, outputs, params=None):
        """Calculate the gradients of the objective function."""
        (theta,) = inputs
        # calculate gradients
        log_prob_grad = -self._beta * self._objective(theta, sensi_orders=(1,))
        outputs[0][0] = log_prob_grad


class PymcSampler(Sampler):
    """Wrapper around Pymc v4 samplers.

    Parameters
    ----------
    step_function:
        A pymc step function, e.g. NUTS, Slice. If not specified, pymc
        determines one automatically (preferable).
    **kwargs:
        Options are directly passed on to `pymc.sample`.
    """

    def __init__(
        self,
        step_function=None,
        post_compute_fval: bool = True,
        **kwargs,
    ):
        super().__init__(kwargs)
        self.step_function = step_function
        self.problem: Problem | None = None
        self.x0: np.ndarray | None = None
        self.trace: pymc.backends.Text | None = None
        self.data: az.InferenceData | None = None

    @classmethod
    def translate_options(cls, options):
        """
        Translate options and fill in defaults.

        Parameters
        ----------
        options:
            Options configuring the sampler.
        """
        if not options:
            options = {"chains": 1}
        return options

    def initialize(self, problem: Problem, x0: np.ndarray):
        """
        Initialize the sampler.

        Parameters
        ----------
        problem:
            The problem for which to sample.
        x0:
            Should, but is not required to, be used as initial parameter.
        """
        self.problem = problem
        if x0 is not None:
            if len(x0) != problem.dim:
                x0 = problem.get_reduced_vector(x0)
        self.x0 = x0
        self.trace = None
        self.data = None

        self.problem.objective.history = MemoryHistory()

    def sample(self, n_samples: int, beta: float = 1.0):
        """
        Sample the problem.

        Parameters
        ----------
        n_samples:
            Number of samples to be computed.
        """
        try:
            import pymc
        except ImportError:
            raise SamplerImportError("pymc") from None

        problem = self.problem
        log_post = PymcObjectiveOp.create_instance(problem.objective, beta)
        trace = self.trace

        x0 = None
        x_names_free = problem.get_reduced_vector(problem.x_names)
        if self.x0 is not None:
            x0 = {
                x_name: val
                for x_name, val in zip(problem.x_names, self.x0)
                if x_name in x_names_free
            }

        # create model context
        with pymc.Model():
            # parameter bounds as uniform prior
            _k = [
                pymc.Uniform(x_name, lower=lb, upper=ub)
                for x_name, lb, ub in zip(
                    x_names_free,
                    problem.lb,
                    problem.ub,
                )
            ]

            # convert parameters to PyTensor tensor variable
            theta = pt.as_tensor_variable(_k)

            # define distribution with log-posterior as density
            pymc.Potential("potential", log_post(theta))

            # record function values
            pymc.Deterministic("loggyposty", log_post(theta))

            # step, by default automatically determined by pymc
            step = None
            if self.step_function:
                step = self.step_function()

            # perform the actual sampling
            data = pymc.sample(
                draws=int(n_samples),
                trace=trace,
                initvals=x0,
                step=step,
                **self.options,
            )

        self.data = data

    def get_samples(self) -> McmcPtResult:
        """Convert result from pymc to McmcPtResult."""
        # dimensions
        n_par, n_chain, n_iter = np.asarray(
            self.data.posterior.to_array()
        ).shape
        n_par -= 1  # remove log-posterior

        # parameters
        trace_x = np.empty(shape=(n_chain, n_iter, n_par))
        par_ids = self.problem.get_reduced_vector(self.problem.x_names)
        if len(par_ids) != n_par:
            raise AssertionError("Mismatch of parameter dimension")
        for i_par, par_id in enumerate(par_ids):
            trace_x[:, :, i_par] = np.asarray(self.data.posterior[par_id])

        # function values
        trace_neglogpost = -np.asarray(self.data.posterior["loggyposty"])

        if (
            trace_x.shape[0] != trace_neglogpost.shape[0]
            or trace_x.shape[1] != trace_neglogpost.shape[1]
            or trace_x.shape[2] != self.problem.dim
        ):
            raise ValueError(
                "Trace dimensions are inconsistent: "
                f"{trace_x.shape=} {trace_neglogpost.shape=} {self.problem.dim=}"
            )

        return McmcPtResult(
            trace_x=np.array(trace_x),
            trace_neglogpost=np.array(trace_neglogpost),
            trace_neglogprior=np.full(trace_neglogpost.shape, np.nan),
            betas=np.array([1.0] * trace_x.shape[0]),
        )
