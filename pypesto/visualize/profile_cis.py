from __future__ import annotations

from collections.abc import Sequence
from typing import Literal
import warnings

import matplotlib as mpl
import matplotlib.axes
import numpy as np
from matplotlib.colors import is_color_like

from ..profile import calculate_approximate_ci, chi2_quantile_to_ratio
from ..result import Result
from .misc import (
    _UNSET,
    _ci_panel_lowlevel,
    ci_panel_size,
    get_ax,
    process_deprecated_kwarg,
)
from ._style import CI_BAR_HEIGHT, resolve_style


def profile_cis(
    result: Result,
    confidence_levels: float | Sequence[float] | None = None,
    df: int = 1,
    profile_indices: Sequence[int] | None = None,
    profile_list: int = 0,
    colors: Sequence | None = None,
    show_bounds: bool = True,
    show_mle: bool = True,
    ax: matplotlib.axes.Axes | None = None,
    orientation: Literal["v", "h"] = "v",
    size: tuple[float, float] | None = None,
    title: str | None = "Profile confidence intervals",
    style_kwargs: dict | None = None,
    confidence_level: float = _UNSET,
    color: str | tuple = _UNSET,
) -> matplotlib.axes.Axes:
    """
    Plot approximate confidence intervals based on profiles.

    Supports one or more confidence levels rendered as nested bars with a
    legend identifying each level.  Uses
    :func:`~pypesto.visualize.misc._ci_panel_lowlevel` for rendering,
    which is shared with :func:`sampling_parameter_cis`.

    Parameters
    ----------
    result:
        The result object after profiling.
    confidence_levels:
        One confidence level (float in (0,1)) or a sequence of them.
        Each is translated to an approximate threshold via
        :func:`pypesto.profile.chi2_quantile_to_ratio`.
    df:
        Degrees of freedom of the chi2 distribution.
    profile_indices:
        Integer indices specifying which profiles to plot.
        Defaults to all indices for which profiles exist.
    profile_list:
        Index of the profile list to be used.
    colors:
        One color per confidence level. If not given, a gradient from
        ``style_kwargs["cmap_ci"]`` is used (lighter = wider CI).
    show_bounds:
        Whether to draw parameter bounds.
    show_mle:
        Whether to mark the MLE (best optimizer result) with a tick on
        each CI bar.
    ax:
        Axes object to use. Default: create a new one.
    orientation:
        ``"v"`` (default): parameter names on the y-axis, value on x-axis.
        ``"h"``: transposed.
    size:
        Figure size ``(width, height)`` in inches; only used when ``ax`` is
        ``None``. When ``None`` it is derived from the parameter count via
        :func:`~pypesto.visualize.misc.ci_panel_size` (the bar axis grows
        with the number of parameters).
    title:
        Axes title. Pass ``None`` to suppress.
    style_kwargs:
        Optional style overrides. Supported keys:

        - ``"cmap_ci"`` – colormap for CI bars (default ``"Blues"``);
          ignored when ``colors`` is provided.
        - ``"bound_color"`` – color of the parameter-bound lines.
        - ``"bound_linestyle"`` – linestyle of the bound lines (default ``"--"``).
        - ``"bound_linewidth"`` – linewidth of the bound lines.
        - ``"bound_alpha"`` – opacity of the bound lines.

    Returns
    -------
    ax:
        The plot axes.
    """
    style = resolve_style(style_kwargs)

    confidence_levels = process_deprecated_kwarg(
        "confidence_levels",
        confidence_levels,
        "confidence_level",
        confidence_level,
    )
    colors = process_deprecated_kwarg("colors", colors, "color", color)

    if confidence_levels is None:
        confidence_levels = 0.95
    if isinstance(confidence_levels, (int, float)):
        confidence_levels = [float(confidence_levels)]
    else:
        confidence_levels = [float(cl) for cl in confidence_levels]

    problem = result.problem
    profile_list_data = result.profile_result.list[profile_list]

    if profile_indices is None:
        profile_indices = [ix for ix, res in enumerate(profile_list_data) if res]

    n_par = len(profile_indices)
    n_cls = len(confidence_levels)
    ws = [(CI_BAR_HEIGHT / n_cls) * i for i in range(1, n_cls + 1)]

    if colors is None:
        cmap = mpl.colormaps[style["cmap_ci"]]
        colors = [cmap(0.3 + 0.6 * w / max(ws)) for w in ws]
    elif is_color_like(colors):
        colors = [colors] * n_cls

    # sort widest CI first; pair with colors before sorting
    levels_colors = sorted(zip(confidence_levels, colors, strict=True), reverse=True)

    # build ci_data: one entry per level, widest first
    ci_data = []
    for level, color in levels_colors:
        confidence_ratio = chi2_quantile_to_ratio(level, df=df)
        lb_arr = np.zeros(n_par)
        ub_arr = np.zeros(n_par)
        for j, i_par in enumerate(profile_indices):
            xs = profile_list_data[i_par].x_path[i_par]
            ratios = profile_list_data[i_par].ratio_path
            lb_arr[j], ub_arr[j] = calculate_approximate_ci(
                xs=xs, ratios=ratios, confidence_ratio=confidence_ratio
            )
        ci_data.append((level, lb_arr, ub_arr, color))

    # MLE point estimates
    point_estimates = None
    if (
        show_mle
        and result.optimize_result is not None
        and len(result.optimize_result.list) > 0
    ):
        best_x = result.optimize_result.list[0].x
        if best_x is not None:
            point_estimates = np.array([best_x[i_par] for i_par in profile_indices])

    lbs = [float(problem.lb_full[i_par]) for i_par in profile_indices]
    ubs = [float(problem.ub_full[i_par]) for i_par in profile_indices]
    x_names = [problem.x_names[ix] for ix in profile_indices]
    parameter_scales = (
        [problem.x_scales[ix] for ix in profile_indices]
        if getattr(problem, "x_scales", None) is not None
        else None
    )

    if size is None:
        size = ci_panel_size(n_par, orientation)
    ax = get_ax(ax, size)

    return _ci_panel_lowlevel(
        ax, ci_data, x_names, parameter_scales, lbs, ubs, style,
        point_estimates=point_estimates,
        point_estimate_label="MLE",
        show_bounds=show_bounds,
        title=title,
        legend_title="Confidence level:",
        orientation=orientation,
    )


def profile_nested_cis(
    result: Result,
    confidence_levels: Sequence[float] = (0.95, 0.9),
    df: int = 1,
    profile_indices: Sequence[int] | None = None,
    profile_list: int = 0,
    colors: Sequence | None = None,
    ax: matplotlib.axes.Axes | None = None,
    orientation: Literal["v", "h"] = "v",
    title: str | None = "Profile confidence intervals",
) -> matplotlib.axes.Axes:
    """Deprecated wrapper for :func:`profile_cis` with multiple levels.

    The ``title`` argument is forwarded as the axes title.
    """
    warnings.warn(
        "`profile_nested_cis` is deprecated; use `profile_cis` with "
        "`confidence_levels` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return profile_cis(
        result=result,
        confidence_levels=confidence_levels,
        df=df,
        profile_indices=profile_indices,
        profile_list=profile_list,
        colors=colors,
        show_bounds=True,
        show_mle=False,
        ax=ax,
        orientation=orientation,
        title=title,
    )
