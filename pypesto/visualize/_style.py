"""
Visual style defaults for pyPESTO plots.

Constants here are the effective defaults for the visual properties pyPESTO
plotters set explicitly: scatter marker geometry, colormaps, reference and
bound line colors, line widths, histogram styling, ….

Font sizes are mostly controlled through matplotlib rcParams (or pyPESTO's
preset via :func:`apply_style`). A few plot-specific font-size controls stay
local where they are part of that plot's layout.

Users customise these defaults per-call via the ``style_kwargs`` dict that
every public plotter accepts.  Pass e.g.
``style_kwargs={"scatter_size": 20, "cmap_fval": "plasma"}`` to override.
Unknown keys raise a ``UserWarning`` immediately so typos surface fast.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import matplotlib as mpl
import matplotlib.axes
import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.figure
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scatter / marker geometry (used by optimization_scatter, sampling_scatter,
# parameters_lowlevel scatter mode, and future scatter-based plotters)
SCATTER_SIZE = 35
SCATTER_ALPHA = 0.85
SCATTER_LINEWIDTHS = 0.6
SCATTER_EDGECOLORS = "white"
SCATTER_ZORDER = 3

# Colormaps for colour-encoded scatter
CMAP_FVAL = "viridis_r"     # objective-value scatter: yellow = best/lowest
CMAP_POSTERIOR = "viridis"  # log-posterior scatter
CMAP_CI = "Blues"           # nested CI bars (profile_cis): lighter = wider/less certain
CMAP_CORRELATION = "coolwarm"  # parameter correlation matrix: diverging, white at 0
CMAP_DISCRETE = "tab10"      # qualitative palette for distinguishing N categories

# Default flat scatter colour — used when no scalar field (fval / log-posterior)
# is available to drive the colormap (e.g. PCA/UMAP projections without colour_by).
SCATTER_COLOR = "#4878d0"   # calm medium blue

# Default line colour — profiles, sampling median, simulation/mapping lines.
LINE_COLOR = "black"

# Background / reference connecting line (waterfall, optimizer history, …)
REF_LINE_COLOR: list[float] = [0.7, 0.7, 0.7, 0.6]

# MLE / best-optimum marker — dot in profiles, tick in profile_cis, tick in
# sampling_parameter_cis (posterior median), best-cluster color in waterfall
# and scatter plots. matplotlib's tab:red — vivid enough to draw the eye
# while remaining a standard scientific-plot red.
MLE_COLOR = "#d62728"

# Singleton/isolated-start cluster swatch, ref-line and outlier indicators.
OUTLIER_COLOR = "#b3b3b3"

# Experimental/measurement data shown alongside model-derived quantities.
DATA_COLOR = "#1f4e79"

# CI range segment linewidth — the thick horizontal bar in profiles showing
# the CI range, and the point-estimate tick in _ci_panel_lowlevel.
CI_LINEWIDTH = 2.5

# Single-axis plots use matplotlib's default figure size (no constant needed).
# Per-panel size for every grid layout (1-D marginals, traces,
# property-vs-index subplots, pairwise scatter matrices). Landscape (wider
# than tall) — long y-labels read better with the extra width.
# Use: size = (GRID_SIZE_PER_COL * num_col, GRID_SIZE_PER_ROW * num_row)
GRID_SIZE_PER_COL = 3.5
GRID_SIZE_PER_ROW = 2.5
# Extra width to add when a colorbar is present so it doesn't steal space
# from the scatter panels (used by optimization_scatter, sampling_scatter).
COLORBAR_WIDTH = 1.0

# Rectangle/fill styling — applied to histogram bars and interval/category
# rectangles. Default edge is a darker shade of the fill blue (ColorBrewer
# Blues family) so the edge reads as the same hue at higher saturation.
RECTANGLE_COLOR = "#9ecae1"
RECTANGLE_EDGECOLOR = "#3182bd"
RECTANGLE_LINEWIDTH = 1.5
RECTANGLE_ALPHA = 0.6

# KDE overlay line width — used by the density helpers in misc.py
# (plot_diagonal_marginal, plot_density_panel) so the curve looks identical
# on a scatter-matrix diagonal and on a standalone histogram panel.
KDE_LINEWIDTH = 2.0

# Marker geometry for line+marker plots (waterfall scatter overlay,
# profile_lowlevel point markers). Distinct from scatter constants above
# because these markers sit on top of a connecting line, not free-floating.
MARKER_SIZE = 24        # scatter area (points²) — for ax.scatter(s=...)
MARKER_LINEWIDTH = 0.6
LINE_MARKER_SIZE = 5.0  # diameter (points) — for ax.plot(markersize=...)

# Trace-line styling — used by optimizer_history and waterfall for the
# continuous connecting lines drawn for each optimizer run.  Distinct from
# the MARKER_* constants above (which control discrete overlay points) and
# from SCATTER_* (which control free-floating scatter plots).
TRACE_LINEWIDTH = 1.2   # connecting line thickness
TRACE_ALPHA = 0.7       # opacity (softens overlap when many traces overlay)
TRACE_MARKER_SIZE = 2.5 # tiny dot on each evaluation step ("." marker)

# Dense MCMC trace scatter (sampling_fval_traces / sampling_parameter_traces).
# These are separate from the sparse SCATTER_* defaults because thousands of
# overlapping markers need no edge and lower alpha to stay readable.
MCMC_SCATTER_ALPHA = 0.5
MCMC_BURNIN_COLOR = "0.6"
MCMC_BURNIN_CUTOFF_COLOR = "#b94a48"

# Parameter-bound rendering — used everywhere lb/ub appear (profile_cis,
# profiles 1d/2d, optimization_scatter, sampling_scatter, sampling_parameter_*,
# ensemble_identifiability, observable_mapping). One canonical look so the
# family stays coherent.
BOUND_LINESTYLE = "--"
BOUND_COLOR = "0.5"
BOUND_LINEWIDTH = 1.4
BOUND_ALPHA = 0.95
# Fractional padding added to axis limits so bound lines drawn at the true
# lb/ub remain visible (and not flush with the axis spine).
BOUND_VIEW_MARGIN = 0.03

# CI / credible-interval bar geometry and alpha — used by
# ensemble_identifiability bars and the shared CI panel renderer.
CI_BAR_HEIGHT = 0.6
CI_ALPHA = 0.85

# Identifiability category colors — live here (not in C.py) because they are
# purely visual and only used by ensemble_identifiability.
COLOR_HIT_NO_BOUNDS = [0.290, 0.478, 0.722, 0.9]    # steel blue  — identifiable
COLOR_HIT_ONE_BOUND = [0.878, 0.557, 0.235, 0.9]    # amber       — caution
COLOR_HIT_BOTH_BOUNDS = [0.627, 0.176, 0.176, 0.9]  # muted red   — non-identifiable

# ---------------------------------------------------------------------------
# Internal defaults dict — must match the public constant names above
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    "scatter_size": SCATTER_SIZE,
    "scatter_alpha": SCATTER_ALPHA,
    "scatter_linewidths": SCATTER_LINEWIDTHS,
    "scatter_edgecolors": SCATTER_EDGECOLORS,
    "scatter_zorder": SCATTER_ZORDER,
    "cmap_fval": CMAP_FVAL,
    "cmap_posterior": CMAP_POSTERIOR,
    "cmap_ci": CMAP_CI,
    "cmap_correlation": CMAP_CORRELATION,
    "cmap_discrete": CMAP_DISCRETE,
    "scatter_color": SCATTER_COLOR,
    "line_color": LINE_COLOR,
    "ref_line_color": REF_LINE_COLOR,
    "mle_color": MLE_COLOR,
    "outlier_color": OUTLIER_COLOR,
    "data_color": DATA_COLOR,
    "ci_linewidth": CI_LINEWIDTH,
    "rectangle_color": RECTANGLE_COLOR,
    "rectangle_edgecolor": RECTANGLE_EDGECOLOR,
    "rectangle_linewidth": RECTANGLE_LINEWIDTH,
    "rectangle_alpha": RECTANGLE_ALPHA,
    "ci_alpha": CI_ALPHA,
    "marker_size": MARKER_SIZE,
    "marker_linewidth": MARKER_LINEWIDTH,
    "line_marker_size": LINE_MARKER_SIZE,
    "trace_linewidth": TRACE_LINEWIDTH,
    "trace_alpha": TRACE_ALPHA,
    "trace_marker_size": TRACE_MARKER_SIZE,
    "mcmc_scatter_alpha": MCMC_SCATTER_ALPHA,
    "mcmc_burnin_color": MCMC_BURNIN_COLOR,
    "mcmc_burnin_cutoff_color": MCMC_BURNIN_CUTOFF_COLOR,
    "bound_color": BOUND_COLOR,
    "bound_linestyle": BOUND_LINESTYLE,
    "bound_linewidth": BOUND_LINEWIDTH,
    "bound_alpha": BOUND_ALPHA,
}

# ---------------------------------------------------------------------------
# resolve_style
# ---------------------------------------------------------------------------


def resolve_style(style_kwargs: dict | None = None) -> dict:
    """Return the effective style dict, merging defaults with caller overrides.

    Parameters
    ----------
    style_kwargs:
        User-supplied overrides.  Unknown keys trigger a ``UserWarning``
        (so typos surface immediately rather than being silently ignored).

    Returns
    -------
    dict
        Merged style dict with all keys from :data:`_DEFAULTS`.
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


# ---------------------------------------------------------------------------
# add_colorbar
# ---------------------------------------------------------------------------


def add_colorbar(
    fig: matplotlib.figure.Figure,
    axes: np.ndarray,
    values: np.ndarray,
    label: str,
    cmap: str | mpl_colors.Colormap = "viridis_r",
    norm: mpl_colors.Normalize | None = None,
) -> mpl_colors.Colorbar:
    """Add a shared colorbar to *fig* anchored to the right of *axes*.

    Parameters
    ----------
    fig:
        The figure that owns *axes*.
    axes:
        2-D array of :class:`matplotlib.axes.Axes` (as returned by
        :func:`~pypesto.visualize.misc.get_axes_array`).  The colorbar
        steals space from the full set.
    values:
        Data values used to set the colorbar range when *norm* is ``None``.
        Ignored if *norm* is provided.
    label:
        Colorbar axis label (e.g. ``"Objective value"``).
    cmap:
        Colormap name or instance.
    norm:
        Optional explicit :class:`~matplotlib.colors.Normalize`.  When
        ``None`` a simple linear norm over ``[values.min(), values.max()]``
        is computed.

    Returns
    -------
    matplotlib.colorbar.Colorbar
    """
    if norm is None:
        norm = mpl_colors.Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    sm = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Target a fixed ~0.25" wide colorbar strip independent of figure size.
    # fraction is relative to the axes-group width, so we derive it from the
    # figure width (axes group ≈ fig_width − COLORBAR_WIDTH).
    fig_width = fig.get_figwidth()
    axes_group_width = max(1.0, fig_width - COLORBAR_WIDTH)
    fraction = 0.25 / axes_group_width
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=fraction, shrink=0.8, pad=0.02)
    set_colorbar_label(cbar, label)
    return cbar


def set_colorbar_label(cbar, label: str, **kwargs) -> None:
    """Apply pyPESTO's canonical colorbar label orientation."""
    cbar.set_label(label, rotation=90, **kwargs)


# ---------------------------------------------------------------------------
# apply_style — opt-in pyPESTO rcParams preset
# ---------------------------------------------------------------------------


def apply_style() -> None:
    """Apply pyPESTO's recommended matplotlib rcParams.

    Sets larger axis/tick labels, removes top/right spines globally,
    styles legends (auto-placed, framed, lightly translucent fill), and enables
    ``constrained_layout`` for sensible panel spacing. This makes every plot
    share the polished look of :func:`profile_lowlevel_2d` without touching
    any individual plotter.

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


# ---------------------------------------------------------------------------
# Bound-line helpers
# ---------------------------------------------------------------------------


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


def draw_bounds_2d(
    ax: matplotlib.axes.Axes,
    lb_x: float,
    ub_x: float,
    lb_y: float,
    ub_y: float,
    *,
    view_margin: bool = True,
    style: dict | None = None,
) -> Line2D:
    """Draw 4 dashed grey bound lines on a 2-D scatter axis.

    Two vertical (at *lb_x*, *ub_x*) and two horizontal (at *lb_y*, *ub_y*).
    Both axes' limits are extended by :data:`BOUND_VIEW_MARGIN` if
    *view_margin* is true.
    """
    draw_bounds_1d(ax, lb_x, ub_x, axis="x", view_margin=view_margin, style=style)
    return draw_bounds_1d(ax, lb_y, ub_y, axis="y", view_margin=view_margin, style=style)


# ---------------------------------------------------------------------------
# Cluster legend
# ---------------------------------------------------------------------------


def cluster_legend_handles_from_data(
    clusters: np.ndarray,
    cluster_size: np.ndarray,
    colors: np.ndarray,
) -> list[Line2D]:
    """Build a per-cluster legend from :func:`~pypesto.util.assign_clusters` output.

    Parameters
    ----------
    clusters:
        Per-run cluster index, as returned by ``assign_clusters``.
    cluster_size:
        Size of each cluster, as returned by ``assign_clusters``.
    colors:
        RGBA color array of shape ``(n_runs, 4)``, as returned by
        ``assign_colors``.  Alpha values are set to 1 in the swatches so
        balance-alpha dimming does not affect the legend.

    Returns
    -------
    list[Line2D]
        Legend handles — one per real cluster (labeled ``"Cluster 1 (best)"``,
        ``"Cluster 2"``, …) plus one ``"Isolated starts"`` entry if any
        singletons exist.  Returns an empty list when every run is isolated
        (no multi-run clusters).
    """
    colors = np.asarray(colors)
    cluster_size = np.asarray(cluster_size)
    clusters = np.asarray(clusters)
    handles = []
    real_cluster_indices = np.where(cluster_size > 1)[0]
    for legend_idx, cluster_idx in enumerate(real_cluster_indices):
        run_idx = int(np.argwhere(clusters == cluster_idx).flatten()[0])
        # force alpha=1 so balance-alpha dimming doesn't bleed into the legend
        swatch = list(colors[run_idx, :3]) + [1.0]
        label = f"Cluster {legend_idx + 1}" + (" (best)" if legend_idx == 0 else "")
        handles.append(Line2D([0], [0], color=swatch, lw=2, label=label))
    if np.any(cluster_size == 1):
        handles.append(
            Line2D(
                [0], [0],
                color=resolve_style({})["outlier_color"],
                lw=2,
                label="Isolated starts",
            )
        )
    return handles


# ---------------------------------------------------------------------------
# Parameter-axis label formatting (scale-aware)
# ---------------------------------------------------------------------------


def format_parameter_axis_labels(
    parameter_names: Sequence[str],
    parameter_scales: Sequence[str] | None,
) -> tuple[list[str], str]:
    """Format per-parameter axis labels and one shared 'value' axis label.

    Mirrors the convention from :func:`pypesto.visualize.parameters`: when
    every parameter shares the same scale the scale is encoded once in the
    'value' axis label (``"Parameter value (log10)"``) and the per-parameter
    labels stay clean. With mixed scales each parameter gets a suffix
    (``"k1 (log10)"``, ``"k2 (lin)"``) and the 'value' label drops the scale.

    Returns ``(per_parameter_labels, value_axis_label)``.
    """
    if parameter_scales is None:
        return list(parameter_names), "Parameter value"
    scales = list(parameter_scales)
    if len(scales) != len(parameter_names):
        return list(parameter_names), "Parameter value"
    unique_scales = set(scales)
    if len(unique_scales) == 1:
        only = next(iter(unique_scales))
        return list(parameter_names), f"Parameter value ({only})"
    return (
        [f"{name} ({scale})" for name, scale in zip(parameter_names, scales)],
        "Parameter value",
    )
