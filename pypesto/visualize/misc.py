from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from numbers import Number

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

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
from ._style import (
    GRID_SIZE_PER_COL,
    GRID_SIZE_PER_ROW,
    draw_bounds_1d,
    resolve_style,
)
from .clust_color import assign_colors_for_list

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
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.

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


def rgba2rgb(fg: RGB_RGBA, bg: RGB_RGBA = None) -> RGB:
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

    Parameters
    ----------
    n_panels:
        Number of panels to arrange.

    Returns
    -------
    nrows, ncols:
        Smallest grid with ``nrows * ncols >= n_panels`` and aspect ratio
        close to square.
    """
    if n_panels < 1:
        raise ValueError("n_panels must be at least 1.")
    nrows = int(np.ceil(np.sqrt(n_panels)))
    ncols = int(np.ceil(n_panels / nrows))
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
        None. If None, matplotlib's default figure size is used.

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


def plot_density_panel(
    ax: matplotlib.axes.Axes,
    values: np.ndarray,
    bins: int | str = "auto",
    bw_method: str = "scott",
    style: dict | None = None,
    *,
    show_hist: bool = True,
    show_kde: bool = True,
    show_rug: bool = True,
    show_bounds: bool = False,
    lb: float | None = None,
    ub: float | None = None,
):
    """Draw a density panel: histogram, KDE overlay, rug marks, and bounds.

    Element styling is read from the resolved *style* dict:

    - histogram bars: ``rectangle_*``
    - KDE line: ``line_color`` / ``linewidth``
    - rug marks: ``dash_color`` / ``dash_linewidth`` / ``dash_markersize`` /
      ``dash_alpha``
    - parameter-bound lines: ``bound_*`` (drawn only when ``show_bounds`` is
      ``True`` and ``lb`` / ``ub`` are finite)

    Sets x-axis limits to the data range with a 5% margin; if bounds are
    drawn, the limits are extended to include them (never shrunk).

    Parameters
    ----------
    ax:
        Axes to draw into.
    values:
        1-D array of data values.
    bins:
        Histogram bins — passed directly to :func:`matplotlib.axes.Axes.hist`.
    bw_method:
        Bandwidth method for :class:`scipy.stats.gaussian_kde`.
    style:
        Pre-resolved visualization style dict, as returned by
        :func:`pypesto.visualize._style.resolve_style`. When ``None``, defaults
        are used.
    show_hist:
        Whether to draw the histogram bars.
    show_kde:
        Whether to draw the KDE line overlay.
    show_rug:
        Whether to draw rug marks along the x-axis.
    show_bounds:
        Whether to draw parameter-bound lines via
        :func:`pypesto.visualize._style.draw_bounds_1d`. Requires ``lb`` and
        ``ub`` to be passed and finite; silently skipped otherwise.
    lb, ub:
        Lower and upper parameter bounds. Used only when ``show_bounds`` is
        ``True``.

    Returns
    -------
    The bound legend handle if bounds were drawn, otherwise ``None``.
    Callers wire this into their per-panel legend.
    """
    from scipy.stats import gaussian_kde

    style = style if style is not None else resolve_style(None)

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    if show_hist:
        ax.hist(
            values,
            bins=bins,
            density=True,
            color=style["rectangle_color"],
            alpha=style["rectangle_alpha"],
            edgecolor=style["rectangle_edgecolor"],
            linewidth=style["rectangle_linewidth"],
        )

    if show_kde and len(values) > 1 and np.std(values) > 0:
        try:
            kde = gaussian_kde(values, bw_method=bw_method)
            # Extend 3 bandwidths beyond the data so the curve tapers to zero.
            bw = kde.factor * np.std(values, ddof=1)
            x_grid = np.linspace(
                values.min() - 3 * bw, values.max() + 3 * bw, 300
            )
            ax.plot(
                x_grid,
                kde(x_grid),
                color=style["line_color"],
                linewidth=style["linewidth"],
            )
        except np.linalg.LinAlgError:
            pass

    if show_rug:
        ax.plot(
            values,
            np.zeros(len(values)),
            marker="|",
            linestyle="none",
            color=style["dash_color"],
            alpha=style["dash_alpha"],
            markersize=style["dash_markersize"],
            markeredgewidth=style["dash_linewidth"],
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=5,
        )

    # Frame the panel tightly to the data (5% margin around the range, or a
    # sensible fallback when the data is constant).
    data_min = float(values.min())
    data_max = float(values.max())
    spread = data_max - data_min
    margin = spread * 0.05 if spread > 0 else max(abs(data_min) * 0.05, 1.0)
    ax.set_xlim(data_min - margin, data_max + margin)

    # Draw parameter bounds (xlim is extended by draw_bounds_1d when bounds
    # fall outside the data frame; never shrunk).
    if (
        show_bounds
        and lb is not None
        and ub is not None
        and np.isfinite(lb)
        and np.isfinite(ub)
    ):
        return draw_bounds_1d(ax, lb, ub, axis="x", style=style)
    return None


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
