"""Latin hypercube sampling."""

import numpy as np

from .base import CheckedStartpoints
from .util import rescale


def latin_hypercube(
    n_starts: int,
    lb: np.ndarray,
    ub: np.ndarray,
    smooth: bool = True,
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


class LatinHypercubeStartpoints(CheckedStartpoints):
    """Generate latin hypercube-sampled startpoints.

    See e.g. https://en.wikipedia.org/wiki/Latin_hypercube_sampling.
    """

    def __init__(
        self,
        use_guesses: bool = True,
        check_fval: bool = False,
        check_grad: bool = False,
        smooth: bool = True,
    ):
        """Initialize.

        Parameters
        ----------
        use_guesses, check_fval, check_grad:
            As in CheckedStartpoints.
        smooth:
            Whether a (uniformly chosen) random starting point within the
            hypercube [i/n_starts, (i+1)/n_starts] should be chosen (True) or
            the midpoint of the interval (False).
        """
        super().__init__(
            use_guesses=use_guesses,
            check_fval=check_fval,
            check_grad=check_grad,
        )
        self.smooth: bool = smooth

    def sample(
        self,
        n_starts: int,
        lb: np.ndarray,
        ub: np.ndarray,
    ) -> np.ndarray:
        """Call function."""
        return latin_hypercube(
            n_starts=n_starts,
            lb=lb,
            ub=ub,
            smooth=self.smooth,
        )
