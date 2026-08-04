"""Pymc v4 Sampler."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

from ..history import MemoryHistory
from ..objective import ObjectiveBase
from ..problem import Problem
from ..result import McmcPtResult
from .sampler import Sampler, SamplerImportError

logger = logging.getLogger(__name__)

# Lazy import of pymc and pytensor
# Check availability once at module load time
_HAS_PYMC = importlib.util.find_spec("pymc") is not None

if _HAS_PYMC:
    import pymc
    import pytensor.tensor as pt

    _PT_OP_BASE = pt.Op
else:
    pymc = None
    pt = None
    _PT_OP_BASE = object

# implementation based on:
# https://www.pymc.io/projects/examples/en/latest/case_studies/blackbox_external_likelihood_numpy.html


def _eval_last_evaluation(
    objective: ObjectiveBase,
    beta: float,
    theta: np.ndarray,
    last_evaluation: dict,
) -> dict:
    """Evaluate log-posterior value and gradient, caching the last result.

    For gradient-based samplers (e.g. NUTS) pymc evaluates the log-posterior
    and its gradient at the same ``theta`` within a single function call. As
    the function value is computed anyway during a sensitivity run, both are
    obtained via a single ``sensi_orders=(0, 1)`` call and cached, so the
    objective (and hence the simulator) is evaluated only once per ``theta``
    instead of once for the value and once for the gradient.

    Parameters
    ----------
    objective:
        Objective function (negative log-likelihood or -posterior).
    beta:
        Inverse temperature (e.g. in parallel tempering).
    theta:
        Parameter vector.
    last_evaluation:
        Mutable dict shared between the value and gradient Op, holding the
        last evaluated parameter vector and the corresponding (scaled) value
        and gradient.

    Returns
    -------
    The (updated) last_evaluation, with keys ``"key"``, ``"fval"`` and ``"grad"``.
    """
    key = theta.tobytes()
    if last_evaluation.get("key") != key:
        fval, grad = objective(theta, sensi_orders=(0, 1))
        last_evaluation["key"] = key
        last_evaluation["fval"] = -beta * np.asarray(fval)
        last_evaluation["grad"] = -beta * np.asarray(grad)
    return last_evaluation


# TODO: once Python 3.11 support is dropped, require only ArviZ >=1.1.0
#  and simplify this helper to `data.posterior.to_dataset()`.
def _get_posterior_dataset(data: Any) -> Any:
    """Return posterior as an xarray Dataset across ArviZ versions."""
    posterior = data.posterior
    if hasattr(posterior, "to_array"):
        return posterior
    return posterior.to_dataset()


class PymcObjectiveOp(_PT_OP_BASE):
    """PyTensor wrapper around a (non-normalized) log-probability function."""

    # Class attributes - set to actual types if pt is available, None otherwise
    # expects a vector of parameter values when called
    itypes = [pt.dvector] if pt is not None else None
    # outputs a single scalar value (the log prob)
    otypes = [pt.dscalar] if pt is not None else None

    @staticmethod
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
        # Check dependencies
        if not _HAS_PYMC:
            raise SamplerImportError("pymc")
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

        # cache shared with the gradient Op, so value and gradient at the same
        # theta require only a single objective (simulator) evaluation
        self._last_evaluation: dict = {}
        self._log_prob_grad = PymcGradientOp(
            objective, beta, self._last_evaluation
        )

    def perform(self, node, inputs, outputs, params=None):
        """Calculate the objective function value (reusing the shared cache)."""
        (theta,) = inputs
        last_evaluation = _eval_last_evaluation(
            self._objective, self._beta, theta, self._last_evaluation
        )
        outputs[0][0] = np.array(last_evaluation["fval"])

    def grad(self, inputs, g):  # noqa
        """Calculate the vector-Jacobian product."""
        # the method that calculates the gradients - it actually returns the
        # vector-Jacobian product - g[0] is a vector of parameter values
        (theta,) = inputs  # our parameters
        return [g[0] * self._log_prob_grad(theta)]


class PymcGradientOp(_PT_OP_BASE):
    """PyTensor wrapper around a (non-normalized) log-probability gradient."""

    # Class attributes - set to actual types if pt is available, None otherwise
    # expects a vector of parameter values when called
    itypes = [pt.dvector] if pt is not None else None
    # outputs a single scalar value (the log prob)
    otypes = [pt.dvector] if pt is not None else None

    def __init__(
        self,
        objective: ObjectiveBase,
        beta: float,
        last_evaluation: dict | None = None,
    ):
        # Check dependencies
        if not _HAS_PYMC:
            raise SamplerImportError("pymc")
        self._objective: ObjectiveBase = objective
        self._beta: float = beta
        # shared with the value Op so that value and gradient at the same theta
        # require only a single objective (simulator) evaluation
        self._last_evaluation: dict = last_evaluation or {}

    def perform(self, node, inputs, outputs, params=None):
        """Calculate the gradients of the objective function."""
        (theta,) = inputs
        # calculate gradients (reusing the shared cache)
        last_evaluation = _eval_last_evaluation(
            self._objective, self._beta, theta, self._last_evaluation
        )
        outputs[0][0] = last_evaluation["grad"]


class PymcSampler(Sampler):
    """Use pymc for sampling.

    Wrapper around Pymc https://www.pymc.io/welcome.html samplers.

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
        # Check dependencies
        if not _HAS_PYMC:
            raise SamplerImportError("pymc")

        super().__init__(kwargs)
        self.step_function = step_function
        self.problem: Problem | None = None
        self.x0: np.ndarray | None = None
        self.trace: pymc.backends.Text | None = None
        self.data: Any | None = None

    @classmethod
    def translate_options(cls, options):
        """
        Translate options and fill in defaults.

        Parameters
        ----------
        options:
            Options configuring the sampler.
        """
        options = dict(options or {})
        options.setdefault("chains", 1)
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
        beta:
            Inverse temperature for tempering (default: 1.0).
        """
        problem = self.problem
        log_post = PymcObjectiveOp.create_instance(problem.objective, beta)
        trace = self.trace

        x0 = None
        x_names_free = problem.get_reduced_vector(problem.x_names)
        if self.x0 is not None:
            x0 = {
                x_name: val
                # FIXME: address https://github.com/ICB-DCM/pyPESTO/issues/1681
                #  and change to strict=True
                for x_name, val in zip(problem.x_names, self.x0, strict=False)
                if x_name in x_names_free
            }

        # create model context
        with pymc.Model():
            # parameter bounds as uniform prior
            _k = [
                pymc.Uniform(x_name, lower=lb, upper=ub)
                for x_name, lb, ub in zip(
                    x_names_free, problem.lb, problem.ub, strict=True
                )
            ]

            # convert parameters to PyTensor tensor variable
            theta = pt.as_tensor_variable(_k)

            # evaluate the log-posterior once and reuse the same node
            log_post_theta = log_post(theta)

            # define distribution with log-posterior as density
            pymc.Potential("potential", log_post_theta)

            # record function values
            pymc.Deterministic("loggyposty", log_post_theta)

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
        posterior = _get_posterior_dataset(self.data)

        # dimensions
        n_par, n_chain, n_iter = np.asarray(posterior.to_array()).shape
        n_par -= 1  # remove log-posterior

        # parameters
        trace_x = np.empty(shape=(n_chain, n_iter, n_par))
        par_ids = self.problem.get_reduced_vector(self.problem.x_names)
        if len(par_ids) != n_par:
            raise AssertionError("Mismatch of parameter dimension")
        for i_par, par_id in enumerate(par_ids):
            trace_x[:, :, i_par] = np.asarray(posterior[par_id])

        # function values
        trace_neglogpost = -np.asarray(posterior["loggyposty"])

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
