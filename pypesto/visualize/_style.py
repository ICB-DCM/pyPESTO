"""
Shared styling helpers for ``pypesto.visualize``.

Cross-cutting pieces used by multiple plotters live here so each plotter
does not reinvent them.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

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
