from __future__ import annotations

import matplotlib.axes
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ..ensemble import Ensemble
from .misc import get_ax
from ._style import (
    CI_BAR_HEIGHT,
    COLOR_HIT_BOTH_BOUNDS,
    COLOR_HIT_NO_BOUNDS,
    COLOR_HIT_ONE_BOUND,
    _bounds_legend_handle,
    draw_bounds_1d,
    resolve_style,
)


def ensemble_identifiability(
    ensemble: Ensemble,
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = (
        "Parameter identifiability\n"
        "(ensemble mean ± 1σ, normalised to parameter bounds)"
    ),
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Visualize identifiability of parameter ensemble.

    Plot an overview about how many parameters hit the parameter bounds based
    on an ensemble of parameters. Confidence intervals/credible ranges are
    computed via the ensemble mean plus/minus 1 standard deviation.
    This highlevel routine expects an ensemble object as input.

    .. warning::

        This plot should be interpreted with care. It shows the spread of
        ensemble members (mean ± 1σ) across all included parameter vectors,
        regardless of their objective function values, and is not a rigorous
        identifiability analysis:

        - Ensembles from optimisation endpoints may contain vectors from
          distinct local optima that lie far outside any confidence region,
          inflating the apparent variance.
        - Ensembles from optimisation history traces span the search space by
          construction and will always appear wide.

        For statistically meaningful identifiability statements, complement
        this plot with uncertainty quantification via profile likelihood
        (:func:`pypesto.visualize.profiles`) or Bayesian sampling
        (:func:`pypesto.visualize.sampling_parameter_cis`).

    Parameters
    ----------
    ensemble:
        ensemble of parameter vectors (from pypesto.ensemble)
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified. Defaults to a height that scales with the number
        of parameters.
    title:
        Axes title. Pass ``None`` to suppress.
    style_kwargs:
        Style overrides. Key used by this function:

        - ``ci_alpha`` — transparency of the mean ± 1σ bars (default 0.85).

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The plot axes.
    """
    id_df = ensemble.check_identifiability()
    ax = ensemble_identifiability_lowlevel(
        id_df, ax=ax, size=size, title=title, style_kwargs=style_kwargs
    )
    return ax


def ensemble_identifiability_lowlevel(
    id_df: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = (
        "Parameter identifiability\n"
        "(ensemble mean ± 1σ, normalised to parameter bounds)"
    ),
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Low-level identifiability routine.

    Plot a horizontal bar chart showing the mean ± 1σ range of each parameter
    across the ensemble, normalized to [0, 1] where 0 = lower bound and
    1 = upper bound. Parameters are colored by identifiability category and
    sorted so the most non-identifiable parameters appear at the top.

    .. warning::

        Interpret with care — does not account for objective function values
        or statistical confidence. See :func:`ensemble_identifiability` for
        full caveats.

    Parameters
    ----------
    id_df:
        DataFrame as returned by ``Ensemble.check_identifiability()``.
        Rows are indexed by parameter name; required columns are
        ``lowerBound``, ``upperBound``, ``ensemble_mean``, ``ensemble_std``,
        ``within lb: 1 std``, ``within ub: 1 std``.
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    title:
        Axes title. Pass ``None`` to suppress.
    style_kwargs:
        Style overrides. Key used by this function:

        - ``ci_alpha`` — transparency of the mean ± 1σ bars (default 0.85).

    Returns
    -------
    ax: matplotlib.axes.Axes
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    n_par = len(id_df)
    if size is None:
        size = (8.0, max(3.0, 0.45 * n_par + 1.5))
    ax = get_ax(ax, size)

    lb = id_df["lowerBound"].values.astype(float)
    ub = id_df["upperBound"].values.astype(float)
    mean = id_df["ensemble_mean"].values.astype(float)
    std = id_df["ensemble_std"].values.astype(float)
    par_names = list(id_df.index)

    # Normalise mean, mean−σ, mean+σ to [lb, ub] → [0, 1]
    par_range = ub - lb
    par_range[par_range == 0] = 1.0  # guard against degenerate bounds
    norm_mean = (mean - lb) / par_range
    norm_lo = np.clip((mean - std - lb) / par_range, 0.0, 1.0)
    norm_hi = np.clip((mean + std - lb) / par_range, 0.0, 1.0)

    lb_hit = ~id_df["within lb: 1 std"].values
    ub_hit = ~id_df["within ub: 1 std"].values

    def _color(i: int) -> str:
        if lb_hit[i] and ub_hit[i]:
            return COLOR_HIT_BOTH_BOUNDS
        if lb_hit[i] or ub_hit[i]:
            return COLOR_HIT_ONE_BOUND
        return COLOR_HIT_NO_BOUNDS

    # Sort: most non-identifiable first (both > one bound > identifiable),
    # then by descending CI width so wider bars rise to the top within a group.
    def _sort_key(i: int):
        n_bounds = int(lb_hit[i]) + int(ub_hit[i])
        return (-n_bounds, -(norm_hi[i] - norm_lo[i]))

    order = sorted(range(n_par), key=_sort_key)

    # y positions: 0 = bottom, n_par-1 = top (most non-identifiable at top)
    bar_height = CI_BAR_HEIGHT
    for plot_idx, par_idx in enumerate(order):
        y = n_par - 1 - plot_idx
        color = _color(par_idx)
        width = norm_hi[par_idx] - norm_lo[par_idx]
        ax.barh(y, width, left=norm_lo[par_idx], height=bar_height,
                color=color, alpha=style["ci_alpha"], zorder=2)
        # White tick at mean position
        ax.plot(
            [norm_mean[par_idx], norm_mean[par_idx]],
            [y - bar_height / 2, y + bar_height / 2],
            color="white", linewidth=1.5, zorder=3,
        )

    # Y-axis: parameter names in visual order (top = order[0])
    tick_labels = [par_names[order[n_par - 1 - k]] for k in range(n_par)]
    ax.set_yticks(range(n_par))
    ax.set_yticklabels(tick_labels)
    ax.set_ylim(-0.7, n_par - 0.3)

    # X-axis: normalised [0, 1] with bound annotations
    draw_bounds_1d(ax, 0.0, 1.0, axis="x", view_margin=False, style=style)
    ax.set_xlim(-0.08, 1.08)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["lb", "0.25", "0.5", "0.75", "ub"])
    ax.set_xlabel("Normalised parameter value  (lb = lower bound, ub = upper bound)")

    if title is not None:
        ax.set_title(title)

    # Legend
    n_identifiable = int(np.sum(~lb_hit & ~ub_hit))
    pct_id = 100.0 * n_identifiable / n_par if n_par > 0 else 0.0
    legend_handles: list = []
    if np.any(~lb_hit & ~ub_hit):
        legend_handles.append(
            Patch(facecolor=COLOR_HIT_NO_BOUNDS, alpha=style["ci_alpha"],
                  label=f"Identifiable  ({n_identifiable}/{n_par},  {pct_id:.0f}%)")
        )
    if np.any((lb_hit | ub_hit) & ~(lb_hit & ub_hit)):
        legend_handles.append(
            Patch(facecolor=COLOR_HIT_ONE_BOUND, alpha=style["ci_alpha"],
                  label="At one bound")
        )
    if np.any(lb_hit & ub_hit):
        legend_handles.append(
            Patch(facecolor=COLOR_HIT_BOTH_BOUNDS, alpha=style["ci_alpha"],
                  label="At both bounds")
        )
    legend_handles.append(_bounds_legend_handle(style=style))
    ax.legend(handles=legend_handles)


    return ax
