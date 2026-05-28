"""
Visual style for ``pypesto.visualize``.

Default constants, the ``style_kwargs`` registry, and small cross-module
helpers.

Users override any default per call via ``style_kwargs``, validated against
:data:`_DEFAULTS`::

    waterfall(result, style_kwargs={"mle_color": "tab:purple"})

Keys are named after the **visual element**, not the plot using them:
``line_*`` (any line), ``dash_*`` (any tick / rug marker), ``rectangle_*``
(fills), ``bound_*`` (parameter bounds). Plot-specific keys (e.g.
``trace_linewidth``) only when a default genuinely diverges.

TODO (capstone): trim the naming note above once the series has settled.
"""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib as mpl
import matplotlib.axes
import numpy as np
from matplotlib.lines import Line2D

# Colors — semantic roles
# -----------------------
MLE_COLOR = "#d62728"      # tab:red — best cluster + MLE markers
OUTLIER_COLOR = "#b3b3b3"  # mid-grey — singleton / outlier starts

# Colormaps
# ---------
CMAP_DISCRETE = "tab10"    # qualitative: cluster + per-variable colours

# Lines (KDE curves, simulation / model-fit lines, …)
# ---------------------------------------------------
LINE_COLOR = "#145685"
LINEWIDTH = 1.5

# Dash markers (rug ticks, CI endpoints, …)
# -----------------------------------------
DASH_COLOR = "#174261"
DASH_LINEWIDTH = 1.2   # markeredgewidth
DASH_MARKERSIZE = 10   # marker length
DASH_ALPHA = 0.8

# Rectangle / histogram fills
# ---------------------------
RECTANGLE_COLOR = "#3182bd"
RECTANGLE_EDGECOLOR = "#000000"
RECTANGLE_LINEWIDTH = 1.0
RECTANGLE_ALPHA = 0.6

# Grid sizing — per-panel inches for multi-panel grids (size=None default)
# ------------------------------------------------------------------------
GRID_SIZE_PER_COL = 3.5
GRID_SIZE_PER_ROW = 2.5

# Parameter bounds
# ----------------
BOUND_LINESTYLE = "--"
BOUND_COLOR = "0.5"
BOUND_LINEWIDTH = 1.4
BOUND_ALPHA = 0.95
BOUND_VIEW_MARGIN = 0.03   # axis-limit padding so bound lines aren't flush with the spine

# Style registry
# --------------

_DEFAULTS: dict[str, object] = {
    "mle_color": MLE_COLOR,
    "outlier_color": OUTLIER_COLOR,
    "cmap_discrete": CMAP_DISCRETE,
    "line_color": LINE_COLOR,
    "linewidth": LINEWIDTH,
    "dash_color": DASH_COLOR,
    "dash_linewidth": DASH_LINEWIDTH,
    "dash_markersize": DASH_MARKERSIZE,
    "dash_alpha": DASH_ALPHA,
    "rectangle_color": RECTANGLE_COLOR,
    "rectangle_edgecolor": RECTANGLE_EDGECOLOR,
    "rectangle_linewidth": RECTANGLE_LINEWIDTH,
    "rectangle_alpha": RECTANGLE_ALPHA,
    "bound_color": BOUND_COLOR,
    "bound_linestyle": BOUND_LINESTYLE,
    "bound_linewidth": BOUND_LINEWIDTH,
    "bound_alpha": BOUND_ALPHA,
}


def resolve_style(style_kwargs: dict | None = None) -> dict:
    """Return the effective style dict, merging defaults with caller overrides.

    Parameters
    ----------
    style_kwargs:
        User-supplied overrides. Unknown keys raise a ``UserWarning`` so
        typos surface immediately.

    Returns
    -------
    dict
        Merged style dict with all keys from :data:`_DEFAULTS`, with
        caller overrides applied on top.
    """
    style = dict(_DEFAULTS)
    if style_kwargs:
        unknown = set(style_kwargs) - set(_DEFAULTS)
        if unknown:
            warnings.warn(
                f"Unknown style_kwargs keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(_DEFAULTS)}.",
                UserWarning,
                stacklevel=3,
            )
        style.update(style_kwargs)
    return style


# rcParams preset, not default, opt-in via ``apply_style()``
# ---------------


def apply_style() -> None:
    """Apply pyPESTO's recommended matplotlib rcParams.

    Sets larger axis/tick labels, removes top/right spines globally,
    styles legends (auto-placed, framed, lightly translucent fill), and enables
    ``constrained_layout`` for sensible panel spacing.

    Opt-in: not called automatically. Users (and pyPESTO's example
    notebooks/docs) call this once at the top of a session.
    """
    mpl.rcParams.update({
        "axes.labelsize": 13,
        "axes.labelweight": mpl.rcParamsDefault["axes.labelweight"],
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": mpl.rcParamsDefault["legend.fontsize"],
        # Legends: auto-placed, framed, and lightly translucent so text reads
        # clearly without making the legend feel heavy.
        "legend.loc": "best",
        "legend.frameon": True,
        "legend.framealpha": 0.6,
        "legend.edgecolor": "0.7",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.constrained_layout.use": True,
    })


# Bound-line helpers
# ------------------


def _bounds_legend_handle(label: str = "Bounds", style: dict | None = None) -> Line2D:
    """Return a Line2D matching the bound style suitable as a legend handle."""
    s = style or {}
    return Line2D(
        [0],
        [0],
        color=s.get("bound_color", BOUND_COLOR),
        linestyle=s.get("bound_linestyle", BOUND_LINESTYLE),
        linewidth=s.get("bound_linewidth", BOUND_LINEWIDTH),
        alpha=s.get("bound_alpha", BOUND_ALPHA),
        label=label,
    )


def draw_bounds_1d(
    ax: matplotlib.axes.Axes,
    lb: float,
    ub: float,
    *,
    axis: Literal["x", "y"] = "x",
    view_margin: bool = True,
    style: dict | None = None,
) -> Line2D:
    """Draw the canonical pyPESTO parameter-bound lines on *ax*.

    ``axis="x"`` draws two vertical dashed lines (``axvline``) at *lb* and
    *ub*; ``axis="y"`` draws two horizontal dashed lines (``axhline``).

    When *view_margin* is true the corresponding axis limits are extended by
    :data:`BOUND_VIEW_MARGIN` * (ub - lb) so the bound lines are visible
    rather than flush with the spine.

    Returns a :class:`~matplotlib.lines.Line2D` that can be passed as a
    legend handle (the lines drawn on the axis are not labeled to keep the
    automatic legend clean).
    """
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    s = style or {}
    color = s.get("bound_color", BOUND_COLOR)
    linestyle = s.get("bound_linestyle", BOUND_LINESTYLE)
    linewidth = s.get("bound_linewidth", BOUND_LINEWIDTH)
    alpha = s.get("bound_alpha", BOUND_ALPHA)
    drawer = ax.axvline if axis == "x" else ax.axhline
    for bound in (lb, ub):
        drawer(bound, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, zorder=1)
    if view_margin and np.isfinite(lb) and np.isfinite(ub) and ub > lb:
        margin = BOUND_VIEW_MARGIN * (ub - lb)
        if axis == "x":
            cur_lo, cur_hi = ax.get_xlim()
            ax.set_xlim(min(cur_lo, lb - margin), max(cur_hi, ub + margin))
        else:
            cur_lo, cur_hi = ax.get_ylim()
            ax.set_ylim(min(cur_lo, lb - margin), max(cur_hi, ub + margin))
    return _bounds_legend_handle(style=s)
