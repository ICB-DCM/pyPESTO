from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from numbers import Number

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from typing import Literal

from ..C import (
    ALL,
    ALL_CLUSTERED,
    COLOR,
    FIRST_CLUSTER,
    FREE_ONLY,
    LEN_RGB,
    LEN_RGBA,
    RGB,
    RGB_RGBA,
    RGBA_ALPHA,
    RGBA_MAX,
    RGBA_MIN,
    RGBA_WHITE,
)
from ..result import Result
from ..util import assign_clusters, delete_nan_inf
from .clust_color import assign_colors_for_list
from ._style import (
    BOUND_VIEW_MARGIN,
    CI_BAR_HEIGHT,
    GRID_SIZE_PER_COL,
    GRID_SIZE_PER_ROW,
    RECTANGLE_ALPHA,
    RECTANGLE_COLOR,
    RECTANGLE_EDGECOLOR,
    RECTANGLE_LINEWIDTH,
    KDE_LINEWIDTH,
    _bounds_legend_handle,
    format_parameter_axis_labels,
)

logger = logging.getLogger(__name__)


def process_result_list(
    results: Result | list[Result],
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    style: dict | None = None,
) -> tuple[list[Result], list[COLOR], list[str]]:
    """
    Assign colors and legends to a list of results, check user provided lists.

    Parameters
    ----------
    results:
        list of pypesto.Result objects or a single pypesto.Result
    colors:
        list of colors recognized by matplotlib, or single color
    legends:
        labels for line plots
    style:
        Resolved pyPESTO visualization style. Forwarded to
        :func:`assign_colors_for_list`. Defaults to the global style.

    Returns
    -------
    results:
       list of pypesto.Result objects
    colors:
        One for each element in 'results'.
    legends:
        labels for line plots
    """
    # check how many results were passed
    single_result = False
    legend_type_error = False
    if isinstance(results, list):
        if len(results) == 1:
            single_result = True
    else:
        single_result = True
        results = [results]

    # handle results according to their number
    if single_result:
        # assign colors and create list for later handling
        if colors is not None and isinstance(colors, list):
            colors = [np.array(colors)]
        else:
            colors = [colors]

        # create list of legends for later handling
        if not isinstance(legends, list):
            legends = [legends]
        try:
            str(legends[0])
        except TypeError:
            legend_type_error = True
    else:
        # if more than one result is passed, we use one color per result
        colors = assign_colors_for_list(len(results), colors, style=style)

        # check whether list of legends has the correct length
        if legends is None:
            # No legends were passed: create some custom legends
            legends = []
            for i_leg in range(len(results)):
                legends.append("Result " + str(i_leg))
        else:
            # legends were passed by user: check length
            try:
                if isinstance(legends, str):
                    legends = [legends]
                if len(legends) != len(results):
                    raise ValueError(
                        "List of results passed and list of labels do "
                        "not have the same length."
                    )
            except TypeError:
                legend_type_error = True

    if legend_type_error:
        raise TypeError("Unexpected legend type.")

    return results, colors, legends


def process_offset_y(
    offset_y: float | None, scale_y: str, min_val: float
) -> float:
    """
    Compute offset for y-axis, depend on user settings.

    Parameters
    ----------
    offset_y:
       value for offsetting the later plotted values, in order to ensure
       positivity if a semilog-plot is used
    scale_y:
       Can be 'lin' or 'log10', specifying whether values should be plotted
       on linear or on log10-scale
    min_val:
        Smallest value to be plotted

    Returns
    -------
    offset_y: float
       value for offsetting the later plotted values, in order to ensure
       positivity if a semilog-plot is used
    """
    # check whether the offset specified by the user is sufficient
    if offset_y is not None:
        if (scale_y == "log10") and (min_val + offset_y <= 0.0):
            warnings.warn(
                "Offset specified by user is insufficient. "
                "Ignoring specified offset and using "
                + str(np.abs(min_val) + 1.0)
                + " instead.",
                stacklevel=2,
            )
        else:
            return offset_y
    else:
        # check whether scaling is lin or log10
        if scale_y == "lin":
            # linear scaling doesn't need any offset
            return 0.0

    return 1.0 - min_val


def process_y_limits(
    ax: matplotlib.axes.Axes,
    y_limits: None | Iterable[float] | np.ndarray,
) -> matplotlib.axes.Axes:
    """
    Apply user specified limits of y-axis.

    Parameters
    ----------
    ax:
        Axes object to use.
    y_limits:
       y_limits, minimum and maximum, for current axes object

    Returns
    -------
    ax:
        Axes object to use.
    """
    # apply y-limits, if they were specified by the user
    if y_limits is not None:
        y_limits = np.array(y_limits)

        # check validity of bounds
        if y_limits.size == 0:
            y_limits = np.array(ax.get_ylim())
        elif y_limits.size == 1:
            # if the user specified only an upper bound
            tmp_y_limits = ax.get_ylim()
            y_limits = [tmp_y_limits[0], y_limits]
        elif y_limits.size > 1:
            y_limits = [y_limits[0], y_limits[1]]

        # check validity of bounds if plotting in log-scale
        if ax.get_yscale() == "log" and y_limits[0] <= 0.0:
            tmp_y_limits = ax.get_ylim()
            if y_limits[1] <= 0.0:
                y_limits = tmp_y_limits
                warnings.warn(
                    "Invalid bounds for plotting in "
                    "log-scale. Using defaults bounds.",
                    stacklevel=2,
                )
            else:
                y_limits = [tmp_y_limits[0], y_limits[1]]
                warnings.warn(
                    "Invalid lower bound for plotting in "
                    "log-scale. Using only upper bound.",
                    stacklevel=2,
                )

        # set limits
        ax.set_ylim(y_limits)

    else:
        # No limits passed, but if we have a result list: check the limits
        ax_limits = np.array(ax.get_ylim())
        data_limits = ax.dataLim.ymin, ax.dataLim.ymax

        # Check if data fits to axes and adapt limits, if necessary
        if np.isfinite(data_limits).all() and (
            ax_limits[0] > data_limits[0] or ax_limits[1] < data_limits[1]
        ):
            # Get range of data
            data_range = data_limits[1] - data_limits[0]
            if ax.get_yscale() == "log":
                data_range = np.log10(data_range)
                new_limits = (
                    np.power(10, np.log10(data_limits[0]) - 0.02 * data_range),
                    np.power(10, np.log10(data_limits[1]) + 0.02 * data_range),
                )
            else:
                new_limits = (
                    data_limits[0] - 0.02 * data_range,
                    data_limits[1] + 0.02 * data_range,
                )

            # set limits
            ax.set_ylim(new_limits)

    return ax


def rgba2rgb(fg: RGB_RGBA, bg: RGB_RGBA | None = None) -> RGB:
    """Combine two colors, removing transparency.

    Parameters
    ----------
    fg:
        Foreground color.
    bg:
        Background color.

    Returns
    -------
    The combined color.
    """
    if bg is None:
        bg = RGBA_WHITE
    if len(bg) == LEN_RGBA:
        # return foreground if background is fully transparent
        if bg[RGBA_ALPHA] == RGBA_MIN:
            return fg
    else:
        if len(bg) != LEN_RGB:
            raise IndexError(
                "A background color of unexpected length was provided: {bg}"
            )
        bg = (*bg, RGBA_MAX)

    # return the foreground color if has no transparency
    if len(fg) == LEN_RGB or fg[RGBA_ALPHA] == RGBA_MAX:
        return fg
    if len(fg) != LEN_RGBA:
        raise IndexError(
            "A foreground color of unexpected length was provided: {fg}"
        )

    def apparent_composite_color_component(
        fg_component: float,
        bg_component: float,
        fg_alpha: float = fg[RGBA_ALPHA],
        bg_alpha: float = bg[RGBA_ALPHA],
    ) -> float:
        """
        Composite a foreground over a background color component.

        Porter and Duff equations are used for alpha compositing.

        Parameters
        ----------
        fg_component:
            The foreground color component.
        bg_component:
            The background color component.
        fg_alpha:
            The foreground color transparency/alpha component.
        bg_alpha:
            The background color transparency/alpha component.

        Returns
        -------
        The component of the new color.
        """
        return (
            fg_component * fg_alpha
            + bg_component * bg_alpha * (RGBA_MAX - fg_alpha)
        ) / (fg_alpha + bg_alpha * (RGBA_MAX - fg_alpha))

    return [
        apparent_composite_color_component(fg[i], bg[i])
        for i in range(LEN_RGB)
    ]


def process_start_indices(
    result: Result,
    start_indices: str | int | Iterable[int] = None,
) -> np.ndarray:
    """
    Process the start_indices.

    Create an array of indices if a number was provided, checks that the indices
    do not exceed the max_index and removes starts with non-finite fval.

    Parameters
    ----------
    start_indices:
        list of indices or int specifying an endpoint of the sequence of
        indices. Furthermore the following strings are possible:
            * 'all', this is the default, using all start indices.
            * 'all_clustered', this includes the best start and all clusters
            with the size > 1.
            * 'first_cluster', includes all starts that belong to the first
            cluster.
    result:
        Result to determine maximum allowed length and/or clusters.
    """
    if start_indices is None:
        start_indices = ALL
    if isinstance(start_indices, str):
        if start_indices == ALL:
            start_indices = np.asarray(range(len(result.optimize_result)))
        elif start_indices == ALL_CLUSTERED:
            clust_ind, clust_size = assign_clusters(
                delete_nan_inf(result.optimize_result.fval)[1]
            )
            # get all clusters that have size >= 2 and cluster of best start:
            clust_gr2 = np.where(clust_size > 2)[0]
            clust_gr2 = (
                np.append(clust_gr2, 0) if 0 not in clust_gr2 else clust_gr2
            )
            start_indices = np.concatenate(
                [np.where(clust_ind == i_clust)[0] for i_clust in clust_gr2]
            )
            start_indices = start_indices
        elif start_indices == FIRST_CLUSTER:
            clust_ind = assign_clusters(
                delete_nan_inf(result.optimize_result.fval)[1]
            )[0]
            start_indices = np.where(clust_ind == 0)[0]
        else:
            raise ValueError(
                f"Permissible values for start_indices are {ALL}, "
                f"{ALL_CLUSTERED}, {FIRST_CLUSTER}, an integer or a "
                f"list of indices. Got {start_indices}."
            )
    # if it is an integer n, select the first n starts
    if isinstance(start_indices, Number):
        start_indices = range(int(start_indices))

    # filter out the indices that exceed the range of possible start indices
    start_indices = [
        start_index
        for start_index in start_indices
        if start_index < len(result.optimize_result)
    ]

    # filter out the indices that are not finite
    start_indices_unfiltered = len(start_indices)
    start_indices = [
        start_index
        for start_index in start_indices
        if np.isfinite(result.optimize_result[start_index].fval)
    ]
    if len(start_indices) != start_indices_unfiltered:
        logger.warning(
            "Some start indices were removed due to inf or nan function values."
        )

    return np.asarray(start_indices, dtype=int)


def process_parameter_indices(
    result: Result,
    parameter_indices: str | Iterable[int] = FREE_ONLY,
) -> list:
    """
    Process the parameter indices, always returning a valid array.

    Create an array of indices depending on the string that is provided. Or
    returns the sequence in case a sequence was provided.

    Parameters
    ----------
    result:
        Result to determine maximum allowed length and/or clusters.
    parameter_indices:
        list of indices or str specifying the desired indices. Default is
        `free_only`. Other option is 'all', which included all estimated
        and fixed parameters.
    """
    if isinstance(parameter_indices, str):
        if parameter_indices == ALL:
            return list(range(0, result.problem.dim_full))
        elif parameter_indices == FREE_ONLY:
            return result.problem.x_free_indices
        else:
            raise ValueError(
                "Permissible values for parameter_indices are "
                f"{ALL}, {FREE_ONLY} or a list of indices."
            )
    return list(parameter_indices)


def make_grid_shape(n_panels: int) -> tuple[int, int]:
    """
    Return a near-square ``(nrows, ncols)`` grid for ``n_panels`` subplots.

    The grid is laid out landscape (``ncols >= nrows``) so panels read left to
    right — e.g. 2 panels give a 1×2 row, 6 panels a 2×3 grid.

    Parameters
    ----------
    n_panels:
        Number of panels to arrange.

    Returns
    -------
    nrows, ncols:
        Smallest grid with ``nrows * ncols >= n_panels`` and aspect ratio
        close to square, never taller than wide.
    """
    if n_panels < 1:
        raise ValueError("n_panels must be at least 1.")
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def get_ax(
    ax: matplotlib.axes.Axes | None = None,
    size: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes:
    """
    Return an Axes, creating one of size ``size`` if ``ax`` is None.

    Parameters
    ----------
    ax:
        Existing matplotlib Axes. If provided, returned unchanged.
    size:
        Figure size ``(width, height)`` in inches; only used when ``ax`` is
        None. When ``None`` matplotlib's default figure size is used.

    Returns
    -------
    ax:
        A matplotlib Axes.
    """
    if ax is not None:
        return ax
    _, ax = plt.subplots(figsize=size, layout="constrained")
    return ax


def get_axes_array(
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
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
        is None. When ``None`` a single panel uses matplotlib's default
        figure size, and a multi-panel grid uses
        ``(GRID_SIZE_PER_COL * ncols, GRID_SIZE_PER_ROW * nrows)``.

    Returns
    -------
    axes:
        A 2-D NumPy object array containing matplotlib Axes.
    """
    if axes is None:
        if size is None and nrows * ncols > 1:
            size = (GRID_SIZE_PER_COL * ncols, GRID_SIZE_PER_ROW * nrows)
        _, axes = plt.subplots(
            nrows,
            ncols,
            squeeze=False,
            figsize=size,
            layout="constrained",
        )
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


def hide_unused_axes(
    axes: np.ndarray,
    n_used: int | None = None,
    used_indices: Iterable[int] | None = None,
    clear: bool = False,
) -> np.ndarray:
    """
    Hide unused axes in a 2-D grid and ensure used axes are visible.

    Parameters
    ----------
    axes:
        2-D NumPy array containing matplotlib Axes.
    n_used:
        Number of leading axes in ``axes.flat`` to keep visible.
    used_indices:
        Flat indices of the axes that should remain visible.
    clear:
        Whether to clear every axis before toggling visibility.

    Returns
    -------
    axes:
        The same 2-D NumPy array with updated visibility.
    """
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim != 2:
        raise ValueError("Pass `axes` as a 2-D NumPy array.")

    if (n_used is None) == (used_indices is None):
        raise ValueError("Pass exactly one of `n_used` or `used_indices`.")

    if used_indices is None:
        if n_used is None or not 0 <= n_used <= axes_array.size:
            raise ValueError(
                f"`n_used` must be between 0 and {axes_array.size}."
            )
        visible_indices = set(range(n_used))
    else:
        visible_indices = set(used_indices)
        invalid_indices = [
            index
            for index in visible_indices
            if index < 0 or index >= axes_array.size
        ]
        if invalid_indices:
            raise ValueError(
                "Pass `used_indices` within the flattened axes range "
                f"[0, {axes_array.size - 1}]."
            )

    for index, ax in enumerate(axes_array.flat):
        if clear:
            ax.clear()
        ax.set_visible(index in visible_indices)

    return axes_array


def plot_diagonal_marginal(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    diag_kind: str = "kde",
    color: str = RECTANGLE_COLOR,
    rectangle_alpha: float = RECTANGLE_ALPHA,
    rectangle_edgecolor: str = RECTANGLE_EDGECOLOR,
    rectangle_linewidth: float = RECTANGLE_LINEWIDTH,
    label: str | None = None,
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
    rectangle_alpha:
        Alpha for histogram bars (``diag_kind="hist"`` only).
    rectangle_edgecolor:
        Edge color for histogram bars (``diag_kind="hist"`` only).
    rectangle_linewidth:
        Edge linewidth for histogram bars (``diag_kind="hist"`` only).
    label:
        Legend label for the plotted artist. If ``None`` no label is set.
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
            ax.plot(x_grid, y_grid, color=color, lw=KDE_LINEWIDTH, label=label)
            ax.set_ylabel("Density")
            return
        except np.linalg.LinAlgError:
            pass

    ax.hist(
        values,
        bins="auto",
        color=color,
        alpha=rectangle_alpha,
        edgecolor=rectangle_edgecolor,
        linewidth=rectangle_linewidth,
        label=label,
    )
    ax.set_ylabel("Count")


def plot_density_panel(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    color=None,
    bins: int | str = "auto",
    bw_method: str = "scott",
    rectangle_color: str = RECTANGLE_COLOR,
    rectangle_alpha: float = RECTANGLE_ALPHA,
    rectangle_edgecolor: str = RECTANGLE_EDGECOLOR,
    rectangle_linewidth: float = RECTANGLE_LINEWIDTH,
    *,
    show_hist: bool = True,
    show_kde: bool = True,
    show_rug: bool = True,
) -> object | None:
    """Draw a density panel: histogram, KDE overlay, and rug marks.

    Parameters
    ----------
    ax:
        Axes to draw into.
    values:
        1-D array of data values.
    color:
        Matplotlib color for all elements. ``None`` uses *rectangle_color*.
    bins:
        Histogram bins — passed directly to :func:`matplotlib.axes.Axes.hist`.
    bw_method:
        Bandwidth method for :class:`scipy.stats.gaussian_kde`.
    rectangle_color, rectangle_alpha, rectangle_edgecolor, rectangle_linewidth:
        Rectangle styling — should come from ``style["rectangle_*"]``.
    show_hist:
        Whether to draw the histogram bars.
    show_kde:
        Whether to draw the KDE line overlay.
    show_rug:
        Whether to draw rug marks along the x-axis.

    Returns
    -------
    The resolved matplotlib color used for all elements, or ``None`` if no
    data was drawn.  Callers use this to build consistent legend proxy handles.
    """
    from scipy.stats import gaussian_kde

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    resolved_color = rectangle_color if color is None else color
    bin_edges = None

    if show_hist:
        _, bin_edges, patches = ax.hist(
            values,
            bins=bins,
            density=True,
            color=resolved_color,
            alpha=rectangle_alpha,
            edgecolor=rectangle_edgecolor,
            linewidth=rectangle_linewidth,
        )
        if resolved_color is None and patches:
            resolved_color = patches[0].get_facecolor()

    if show_kde and len(values) > 1 and np.std(values) > 0:
        try:
            kde = gaussian_kde(values, bw_method=bw_method)
            # Extend 3 bandwidths beyond the data so the curve tapers to zero.
            bw = kde.factor * np.std(values, ddof=1)
            x_grid = np.linspace(values.min() - 3 * bw, values.max() + 3 * bw, 300)
            ax.plot(
                x_grid, kde(x_grid), color=resolved_color, linewidth=KDE_LINEWIDTH
            )
        except np.linalg.LinAlgError:
            pass

    if show_rug and len(values) > 0:
        ax.plot(
            values,
            np.zeros(len(values)),
            marker="|",
            linestyle="none",
            color=resolved_color,
            alpha=0.9,
            markersize=10,
            markeredgewidth=1.2,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=5,
        )

    return resolved_color


def ci_panel_size(
    n_par: int,
    orientation: Literal["v", "h"] = "v",
) -> tuple[float, float]:
    """Return the default figure size for a CI-bar panel.

    Shared by :func:`~pypesto.visualize.profile_cis` and
    :func:`~pypesto.visualize.sampling_parameter_cis`. A CI panel draws one
    bar per parameter, so the dimension along the bars must grow with
    ``n_par``; the other dimension is fixed.

    Layout (for ``orientation="v"``: parameters stacked on the y-axis):

    * width  — fixed ``8.0`` inches
    * height — ``0.5`` inch per parameter bar + ``2.0`` inches for the
      title, axis labels and legend, with a ``3.0`` inch floor so a
      one-parameter plot is not unreasonably short

    ``orientation="h"`` is the transpose (parameters along the x-axis).
    """
    along = max(3.0, 0.5 * n_par + 2.0)
    fixed = 8.0
    return (fixed, along) if orientation == "v" else (along, fixed)


def _ci_panel_lowlevel(
    ax: matplotlib.axes.Axes,
    ci_data: list[tuple],
    parameter_names: list[str],
    parameter_scales: list[str] | None,
    lbs: list[float],
    ubs: list[float],
    style: dict,
    *,
    point_estimates: np.ndarray | None = None,
    point_estimate_label: str = "Estimate",
    show_bounds: bool = True,
    title: str | None = None,
    legend_title: str = "Confidence level:",
    orientation: Literal["v", "h"] = "v",
) -> matplotlib.axes.Axes:
    """
    Shared CI bar renderer for profile_cis and sampling_parameter_cis.

    Parameters
    ----------
    ax:
        Target axes.
    ci_data:
        List of ``(level, lb_per_par, ub_per_par, color)`` tuples, sorted
        widest CI first. ``lb_per_par`` and ``ub_per_par`` are arrays of
        length ``n_par``.
    parameter_names:
        Parameter name per row.
    parameter_scales:
        Scale string (``"log"``, ``"lin"``, …) per parameter, or ``None``.
    lbs:
        Parameter lower bounds.
    ubs:
        Parameter upper bounds.
    style:
        Resolved style dict from :func:`~pypesto.visualize._style.resolve_style`.
    point_estimates:
        Optional array of length ``n_par``. Drawn as a short tick at the
        given x-value (e.g. MLE or posterior median).
    point_estimate_label:
        Legend label for the point-estimate tick.
    show_bounds:
        Whether to draw parameter-bound lines.
    title:
        Axes title. Omitted when ``None``.
    legend_title:
        Title shown above the legend.
    orientation:
        ``"v"`` (default): parameters on y-axis, values on x-axis.
        ``"h"``: transposed.
    """
    n_par = len(parameter_names)
    n_cls = len(ci_data)
    # bar heights: smallest for widest CI (drawn first/underneath),
    # largest for narrowest CI (drawn last/on top)
    ws = [(CI_BAR_HEIGHT / n_cls) * i for i in range(1, n_cls + 1)]

    legend_handles = []

    for i, (level, lb_arr, ub_arr, color) in enumerate(ci_data):
        for j in range(n_par):
            w = float(ub_arr[j]) - float(lb_arr[j])
            lo = float(lb_arr[j])
            if orientation == "v":
                ax.barh(j, width=w, left=lo, height=ws[i],
                        align="center", color=color, edgecolor="none")
            else:
                ax.bar(j, height=w, bottom=lo, width=ws[i],
                       align="center", color=color, edgecolor="none")
        legend_handles.append(Patch(color=color, label=f"{level * 100:g}%"))

    if point_estimates is not None:
        half_h = ws[-1] / 2
        for j, est in enumerate(point_estimates):
            if not np.isfinite(est):
                continue
            if orientation == "v":
                ax.plot([est, est], [j - half_h, j + half_h],
                        color=style["mle_color"], linewidth=style["ci_linewidth"],
                        solid_capstyle="round")
            else:
                ax.plot([j - half_h, j + half_h], [est, est],
                        color=style["mle_color"], linewidth=style["ci_linewidth"],
                        solid_capstyle="round")
        legend_handles.append(
            Line2D([0], [0], color=style["mle_color"],
                   linewidth=style["ci_linewidth"], label=point_estimate_label)
        )

    if show_bounds:
        # Slightly wider than max CI bar half-height so bounds frame the bars.
        half_b = CI_BAR_HEIGHT * 2 / 3
        # One LineCollection, not many ax.plot lines: the legend's loc="best"
        # then ignores the bounds and can still find a clean corner.
        segments = []
        for j in range(n_par):
            if orientation == "v":
                segments.append([(lbs[j], j - half_b), (lbs[j], j + half_b)])
                segments.append([(ubs[j], j - half_b), (ubs[j], j + half_b)])
                if j < n_par - 1:
                    segments.append(
                        [(lbs[j], j + half_b), (lbs[j + 1], j + 1 - half_b)]
                    )
                    segments.append(
                        [(ubs[j], j + half_b), (ubs[j + 1], j + 1 - half_b)]
                    )
            else:
                segments.append([(j - half_b, lbs[j]), (j + half_b, lbs[j])])
                segments.append([(j - half_b, ubs[j]), (j + half_b, ubs[j])])
                if j < n_par - 1:
                    segments.append(
                        [(j + half_b, lbs[j]), (j + 1 - half_b, lbs[j + 1])]
                    )
                    segments.append(
                        [(j + half_b, ubs[j]), (j + 1 - half_b, ubs[j + 1])]
                    )
        ax.add_collection(LineCollection(
            segments,
            linestyles=style["bound_linestyle"],
            colors=style["bound_color"],
            linewidths=style["bound_linewidth"],
            alpha=style["bound_alpha"],
            zorder=1,
        ))
        legend_handles.append(_bounds_legend_handle(style=style))

    formatted_names, value_axis_label = format_parameter_axis_labels(
        parameter_names, parameter_scales
    )
    parameters_ind = np.arange(n_par)
    if orientation == "v":
        ax.set_yticks(parameters_ind)
        ax.set_yticklabels(formatted_names)
        ax.set_ylabel("Parameter")
        ax.set_xlabel(value_axis_label)
    else:
        ax.set_xticks(parameters_ind)
        ax.set_xticklabels(formatted_names, ha="right", rotation=30)
        ax.set_xlabel("Parameter")
        ax.set_ylabel(value_axis_label)

    # Margin on the value axis — bar charts set a sticky edge at the bar base
    # that otherwise leaves the outermost bar flush with the spine.
    span_vals: list[float] = []
    for _level, lb_arr, ub_arr, _color in ci_data:
        span_vals += [float(v) for v in np.asarray(lb_arr)]
        span_vals += [float(v) for v in np.asarray(ub_arr)]
    if point_estimates is not None:
        span_vals += [float(e) for e in np.asarray(point_estimates)]
    if show_bounds:
        span_vals += [float(v) for v in lbs] + [float(v) for v in ubs]
    span_vals = [v for v in span_vals if np.isfinite(v)]
    if span_vals:
        vmin, vmax = min(span_vals), max(span_vals)
        pad = BOUND_VIEW_MARGIN * (vmax - vmin) if vmax > vmin else 0.5
        if orientation == "v":
            ax.set_xlim(vmin - pad, vmax + pad)
        else:
            ax.set_ylim(vmin - pad, vmax + pad)

    if title is not None:
        ax.set_title(title)

    ax.legend(title=legend_title, handles=legend_handles)
    return ax


#: Sentinel meaning "this kwarg was not passed at all."
#: Use as the default for deprecated kwargs so that an explicit
#: ``f(old_kwarg=None)`` can be detected and warned about.
_UNSET = object()


def process_deprecated_kwarg(
    canonical_name: str | None,
    canonical_value,
    deprecated_name: str,
    deprecated_value=_UNSET,
    stacklevel: int = 3,
    note: str | None = None,
):
    """
    Resolve a kwarg that has been renamed or removed.

    The deprecated kwarg must use :data:`_UNSET` as its default in the
    calling function so that an explicit ``f(old_kwarg=None)`` is correctly
    detected and warned about.

    Two modes:

    - **Rename** (``canonical_name`` is given): returns the canonical value
      if the deprecated kwarg was not passed, the deprecated value (with a
      ``DeprecationWarning``) if only the old name was used, or raises
      ``ValueError`` if both are given.
    - **Removal** (``canonical_name`` is ``None``): emits a
      ``DeprecationWarning`` if the deprecated kwarg was passed and returns
      ``None``. ``canonical_value`` is ignored.

    Parameters
    ----------
    canonical_name:
        Name of the canonical (new) kwarg, used in messages. Pass ``None``
        to indicate the kwarg is removed without replacement.
    canonical_value:
        Value passed under the canonical name (or ``None``). Ignored when
        ``canonical_name`` is ``None``.
    deprecated_name:
        Name of the deprecated (old) kwarg, used in messages.
    deprecated_value:
        Value passed under the deprecated name; defaults to :data:`_UNSET`.
    stacklevel:
        Forwarded to :func:`warnings.warn`. Default 3 attributes the
        warning to the caller of the public function that invoked this
        helper.
    note:
        Optional additional sentence appended to the deprecation message
        (e.g. explaining where the behaviour moved to). Useful for the
        removal case.

    Returns
    -------
    value:
        The resolved value, or ``None`` if neither was given (or in the
        removal case).
    """
    if canonical_name is None:
        if deprecated_value is _UNSET:
            return None
        message = f"`{deprecated_name}` is deprecated and has no effect."
        if note:
            message = f"{message} {note}"
        warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
        return None

    if deprecated_value is _UNSET:
        return canonical_value
    if canonical_value is not None:
        raise ValueError(
            f"Pass either `{canonical_name}` or the deprecated "
            f"`{deprecated_name}`, not both."
        )
    message = (
        f"`{deprecated_name}` is deprecated; use `{canonical_name}` instead."
    )
    if note:
        message = f"{message} {note}"
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel)
    return deprecated_value
