import logging
from typing import Optional

import numpy as np
import pytensor.tensor as pt

from ..sample.pymc import PymcObjectiveOp, PymcSampler
from ..sample.sampler import SamplerImportError

logger = logging.getLogger(__name__)


# implementation based on the pymc sampler code in pypesto and:
# https://www.pymc.io/projects/examples/en/latest/variational_inference/variational_api_quickstart.html


class PymcVariational(PymcSampler):
    """Wrapper around Pymc v4 variational inference.

    Parameters
    ----------
    step_function:
        A pymc step function, e.g. NUTS, Slice. If not specified, pymc
        determines one automatically (preferable).
    **kwargs:
        Options are directly passed on to `pymc.fit`.
    """

    def fit(
        self,
        n_iterations: int,
        method: str = 'advi',
        random_seed: Optional[int] = None,
        start_sigma: Optional = None,
        inf_kwargs: Optional = None,
        beta: float = 1.0,
        **kwargs,
    ):
        """
        Sample the problem.

        Parameters
        ----------
        n_iterations:
            Number of iterations.
        method: str or :class:`Inference` of pymc
            string name is case-insensitive in:
            -   'advi'  for ADVI
            -   'fullrank_advi'  for FullRankADVI
            -   'svgd'  for Stein Variational Gradient Descent
            -   'asvgd'  for Amortized Stein Variational Gradient Descent
        random_seed: int
            random seed for reproducibility
        start_sigma: `dict[str, np.ndarray]`
            starting standard deviation for inference, only available for method 'advi'
        inf_kwargs: dict
            additional kwargs passed to pymc.Inference
        beta:
            Inverse temperature (e.g. in parallel tempering).
        """
        try:
            import pymc
        except ImportError:
            raise SamplerImportError("pymc")

        problem = self.problem
        log_post = PymcObjectiveOp.create_instance(problem.objective, beta)

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

            # perform the actual sampling
            data = pymc.fit(
                n=int(n_iterations),
                method=method,
                random_seed=random_seed,
                start=x0,
                start_sigma=start_sigma,
                inf_kwargs=inf_kwargs,
                **kwargs,
            )

        self.data = data

    def sample(self, n_samples: int, beta: float = 1.0):
        """
        Sample from the variational approximation.

        Parameters
        ----------
        n_samples:
            Number of samples to be computed.
        """
        # get InferenceData object
        pymc_data = self.data.sample(n_samples)
        x_names_free = self.problem.get_reduced_vector(self.problem.x_names)
        post_samples = np.concatenate(
            [pymc_data.posterior[name].values for name in x_names_free]
        )
        samples = {
            "trace_x": post_samples,
            "trace_neglogpost": pymc_data.posterior.loggyposty,
            'trace_neglogprior': None,
            'burn_in': 0,
            'time': None,
            'auto_correlation': 0,
            'effective_sample_size': n_samples,
            'message': 'variational inference results',
        }
        return samples

    def save_internal_object(self, filename: str):
        """
        Save the internal pymc object to a file.

        Parameters
        ----------
        filename:
            The file to save to.
        """
        self.data.to_netcdf(filename)

    def load_internal_object(self, filename: str):
        """
        Load the internal pymc object from a file.

        Parameters
        ----------
        filename:
            The file to load from.
        """
        try:
            import arviz
        except ImportError:
            raise ImportError("arviz")
        self.data = arviz.from_netcdf(filename)
