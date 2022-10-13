"""Pymc v4 Sampler."""
from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Union

import numpy as np

from ..history import MemoryHistory
from ..problem import Problem
from ..result import McmcPtResult
from .sampler import Sampler, SamplerImportError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        import aesara.tensor as at
        import arviz as az
        import pymc

        from ..objective.aesara import AesaraObjectiveRV
    except ImportError:
        raise ImportError(
            "Using the pymc sampler requires an installation of the "
            "python packages aesara, arviz and pymc. Please install "
            "these packages via `pip install aesara arviz pymc`."
        )


class AesaraDist(pymc.NoDistribution):
    """PyMC distribution wrapper for AesaraObjectiveRVs."""

    def __new__(
        cls,
        name: str,
        rv_op: AesaraObjectiveRV,
        parameters: at.TensorVariable,
    ):
        """
        Instantiate a PyMC distribution from an AesaraObjectiveRV.

        Parameters
        ----------
        name:
            Name of the distribution
        rv_op:
            Aesara objective random variable
        parameters:
            Input parameters to the aesara objective
        """
        cls.rv_op = rv_op
        return super().__new__(cls, name, [parameters], observed=0)


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

    def __init__(self, step_function=None, **kwargs):
        super().__init__(kwargs)
        self.step_function = step_function
        self.problem: Union[Problem, None] = None
        self.x0: Union[np.ndarray, None] = None
        self.trace: Union[pymc.backends.Text, None] = None
        self.data: Union[az.InferenceData, None] = None
        warnings.warn("The pymc sampler is currently not supported")

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
            options = {'chains': 1}
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

    def sample(self, n_samples: int, coeff: float = 1.0):
        """
        Sample the problem.

        Parameters
        ----------
        n_samples:
            Number of samples to be computed.
        coeff:
            Inverse temperature for the log probability function.
        """
        try:
            import pymc
        except ImportError:
            raise SamplerImportError("pymc")

        problem = self.problem
        log_post_rv = AesaraObjectiveRV(problem.objective, coeff)
        trace = self.trace

        x0 = None
        if self.x0 is not None:
            x0 = {
                x_name: val
                for x_name, val in zip(self.problem.x_names, self.x0)
            }

        # create model context
        with pymc.Model():
            # uniform bounds
            k = [
                pymc.Uniform(x_name, lower=lb, upper=ub)
                for x_name, lb, ub in zip(
                    problem.get_reduced_vector(problem.x_names),
                    problem.lb,
                    problem.ub,
                )
            ]

            # convert parameters to aesara tensor variable
            theta = at.as_tensor_variable(k)

            # define distribution with log-posterior as density
            AesaraDist('log_post', log_post_rv, theta)

            # step, by default automatically determined by pymc
            step = None
            if self.step_function:
                step = self.step_function()

            # perform the actual sampling
            data = pymc.sample(
                draws=int(n_samples),
                trace=trace,
                start=x0,
                step=step,
                **self.options,
            )

        self.data = data

    def get_samples(self) -> McmcPtResult:
        """Convert result from pymc to McmcPtResult."""
        # parameter values
        trace_x = np.asarray(self.data.posterior.to_array()).transpose(
            (1, 2, 0)
        )

        # TODO this is only the negative objective values
        trace_neglogpost = np.asarray(self.data.log_likelihood.to_array())
        # remove trailing dimensions
        trace_neglogpost = np.reshape(
            trace_neglogpost, trace_neglogpost.shape[1:-1]
        )
        # flip sign
        trace_neglogpost = -trace_neglogpost

        if (
            trace_x.shape[0] != trace_neglogpost.shape[0]
            or trace_x.shape[1] != trace_neglogpost.shape[1]
            or trace_x.shape[2] != self.problem.dim
        ):
            raise ValueError("Trace dimensions are inconsistent")

        return McmcPtResult(
            trace_x=np.array(trace_x),
            trace_neglogpost=np.array(trace_neglogpost),
            trace_neglogprior=np.full(trace_neglogpost.shape, np.nan),
            betas=np.array([1.0] * trace_x.shape[0]),
        )
