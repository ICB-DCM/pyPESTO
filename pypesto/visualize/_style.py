"""
Shared styling helpers for ``pypesto.visualize``.

Cross-cutting pieces used by multiple plotters live here so each plotter
does not reinvent them.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_ax(
    ax: Axes | None = None,
    size: tuple[float, float] | None = None,
) -> Axes:
    """
    Return an Axes, creating one of size ``size`` if ``ax`` is None.

    Parameters
    ----------
    ax:
        Existing matplotlib Axes. If provided, returned unchanged.
    size:
        Figure size ``(width, height)`` in inches; only used when ``ax`` is
        None. If None, matplotlib's default figure size is used.

    Returns
    -------
    ax:
        A matplotlib Axes.
    """
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=size)
    return ax


def get_axes_array(
    axes: Axes | np.ndarray | None = None,
    nrows: int = 1,
    ncols: int = 1,
    size: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Return a 2-D array of Axes, creating one if ``axes`` is None.

    Parameters
    ----------
    axes:
        Existing matplotlib Axes grid. If provided, it is normalized to a
        2-D object array and validated against ``(nrows, ncols)``.
    nrows, ncols:
        Expected grid shape.
    size:
        Figure size ``(width, height)`` in inches; only used when ``axes``
        is None.

    Returns
    -------
    axes:
        A 2-D NumPy object array containing matplotlib Axes.
    """
    if axes is None:
        _, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=size)
        return axes

    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 0:
        axes_array = axes_array.reshape(1, 1)
    elif axes_array.ndim == 1:
        if nrows == 1:
            axes_array = axes_array.reshape(1, ncols)
        elif ncols == 1:
            axes_array = axes_array.reshape(nrows, 1)
        else:
            raise ValueError(f"Pass `axes` with shape ({nrows}, {ncols}).")

    if axes_array.shape != (nrows, ncols):
        raise ValueError(f"Pass `axes` with shape ({nrows}, {ncols}).")

    return axes_array


def plot_diagonal_marginal(
    ax: Axes,
    values: np.ndarray,
    diag_kind: str = "kde",
    color: str = "C0",
) -> None:
    """
    Plot a 1-D marginal on a diagonal scatter-matrix panel.

    Parameters
    ----------
    ax:
        Axes to draw into.
    values:
        One-dimensional sample values.
    diag_kind:
        Marginal visualization mode: ``"kde"`` or ``"hist"``.
    color:
        Base matplotlib color for the marginal.
    """
    from scipy.stats import gaussian_kde

    values = np.asarray(values)
    if values.size == 0:
        return
    data_range = values.max() - values.min()
    if data_range == 0:
        data_range = max(abs(float(values.mean())) * 0.1, 0.1)
    x_pad = data_range * 0.25
    x_grid = np.linspace(values.min() - x_pad, values.max() + x_pad, 300)

    if diag_kind == "kde" and len(values) > 1:
        try:
            kde = gaussian_kde(values)
            y_grid = kde(x_grid)
            ax.fill_between(x_grid, y_grid, alpha=0.35, color=color)
            ax.plot(x_grid, y_grid, color=color, lw=1.5)
            ax.set_ylabel("Density")
            return
        except np.linalg.LinAlgError:
            pass

    ax.hist(values, bins="auto", color=color, alpha=0.6)
    ax.set_ylabel("Count")


def process_deprecated_kwarg(
    canonical_name: str,
    canonical_value,
    deprecated_name: str,
    deprecated_value,
    stacklevel: int = 3,
):
    """
    Resolve a kwarg that has been renamed.

    Returns the canonical value if given, the deprecated value (with a
    ``DeprecationWarning``) if only the old name was used, or ``None`` if
    neither was given. Raises ``ValueError`` if both are given.

    Parameters
    ----------
    canonical_name:
        Name of the canonical (new) kwarg, used in messages.
    canonical_value:
        Value passed under the canonical name (or ``None``).
    deprecated_name:
        Name of the deprecated (old) kwarg, used in messages.
    deprecated_value:
        Value passed under the deprecated name (or ``None``).
    stacklevel:
        Forwarded to :func:`warnings.warn`. Default 3 attributes the
        warning to the caller of the public function that invoked this
        helper.

    Returns
    -------
    value:
        The resolved value, or ``None`` if neither was given.
    """
    if deprecated_value is None:
        return canonical_value
    if canonical_value is not None:
        raise ValueError(
            f"Pass either `{canonical_name}` or the deprecated "
            f"`{deprecated_name}`, not both."
        )
    warnings.warn(
        f"`{deprecated_name}` is deprecated; use `{canonical_name}` instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    return deprecated_value
