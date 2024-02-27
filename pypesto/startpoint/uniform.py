"""Uniform sampling."""

import numpy as np

from .base import FunctionStartpoints
from .util import rescale


def uniform(
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


class UniformStartpoints(FunctionStartpoints):
    """Generate uniformly sampled startpoints."""

    def __init__(
        self,
        use_guesses: bool = True,
        check_fval: bool = False,
        check_grad: bool = False,
    ):
        super().__init__(
            function=uniform,
            use_guesses=use_guesses,
            check_fval=check_fval,
            check_grad=check_grad,
        )
