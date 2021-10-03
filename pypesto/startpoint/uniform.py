"""Uniform sampling."""

import numpy as np

from .util import rescale
from .base import StartpointMethod
from ..objective import ObjectiveBase


def _uniform(
    n_starts: int,
    lb: np.ndarray,
    ub: np.ndarray,
) -> np.ndarray:
    """Generate uniform points.

    Parameters
    ----------
    n_starts: Number of starts.
    lb: Lower bound.
    ub: Upper bound.

    Returns
    -------
    xs: Uniformly sampled points in [lb, ub], shape (n_starts, n_x).
    """
    if not np.isfinite(ub).all() or not np.isfinite(lb).all():
        raise ValueError(
            "Cannot use uniform startpoint method with non-finite boundaries.",
        )

    # parse
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))

    # create uniform points in [0, 1]
    xs = np.random.random((n_starts, dim))

    # re-scale
    xs = rescale(xs, lb, ub)

    return xs


class UniformStartpoints(StartpointMethod):
    """Generate uniformly sampled startpoints."""

    def __call__(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: ObjectiveBase = None,
        x_guesses: np.ndarray = None,
    ) -> np.ndarray:
        return _uniform(n_starts=n_starts, lb=lb, ub=ub)


# convenience and legacy
uniform = UniformStartpoints()
