import numpy as np

from .util import rescale
from .base import StartpointMethod
from ..objective import ObjectiveBase


def _latin_hypercube(
    n_starts: int,
    lb: np.ndarray,
    ub: np.ndarray,
    smooth: bool,
) -> np.ndarray:
    """Generate latin hypercube points.

    Parameters
    ----------
    n_starts:
        Number of points.
    lb:
        Lower bound.
    ub:
        Upper bound.
    smooth:
        Whether a (uniformly chosen) random starting point within the
        hypercube [i/n_starts, (i+1)/n_starts] should be chosen (True) or
        the midpoint of the interval (False).

    Returns
    -------
    xs:
        Latin hypercube points, shape (n_starts, n_x).
    """
    if not np.isfinite(ub).all() or not np.isfinite(lb).all():
        raise ValueError(
            "Cannot use latin hypercube startpoint method with non-finite "
            "boundaries.",
        )

    # parse
    dim = lb.size
    lb = lb.reshape((1, -1))
    ub = ub.reshape((1, -1))

    # sample
    xs = _latin_hypercube_unit(n_starts, dim, smooth)

    # re-scale
    xs = rescale(xs, lb, ub)

    return xs


def _latin_hypercube_unit(
    n_starts: int,
    dim: int,
    smooth: bool,
) -> np.ndarray:
    """Generate simple latin hypercube points in [0, 1].

    Parameters are as for `latin_hypercube`.

    Returns
    -------
    xs: Latin hypercube points sampled in [0, 1], shape (n_starts, dim).
    """
    xs = np.empty((n_starts, dim))

    for i_dim in range(dim):
        xs[:, i_dim] = np.random.permutation(np.arange(n_starts))

    if smooth:
        xs += np.random.random((n_starts, dim))
    else:
        xs += 0.5

    xs /= n_starts

    return xs


class LatinHypercubeStartpoints(StartpointMethod):
    """Generate latin hypercube-sampled startpoints.

    See e.g. https://en.wikipedia.org/wiki/Latin_hypercube_sampling."""

    def __init__(
        self,
        smooth: bool = True,
    ):
        """
        Parameters
        ----------
        smooth:
            Whether a (uniformly chosen) random starting point within the
            hypercube [i/n_starts, (i+1)/n_starts] should be chosen (True) or
            the midpoint of the interval (False).
        """
        self.smooth: bool = smooth

    def __call__(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
        objective: ObjectiveBase = None,
        x_guesses: np.ndarray = None,
    ) -> np.ndarray:
        return _latin_hypercube(
            n_starts=n_starts,
            lb=lb,
            ub=ub,
            smooth=self.smooth,
        )


# convenience and legacy
latin_hypercube = LatinHypercubeStartpoints(smooth=True)
