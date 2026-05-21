from __future__ import annotations

import textwrap
from collections.abc import Iterable, Sequence

import matplotlib.axes
import numpy as np
from matplotlib.colors import is_color_like

from pypesto.util import assign_clusters, delete_nan_inf

from ..C import COLOR
from ..result import Result
from .clust_color import assign_colors, assign_colors_for_list
from .misc import (
    get_ax,
    get_axes_array,
    hide_unused_axes,
    make_grid_shape,
    process_result_list,
    process_start_indices,
)
from ._style import (
    cluster_legend_handles_from_data,
    resolve_style,
)


def optimization_run_properties_one_plot(
    results: Result | list[Result],
    properties_to_plot: list[str] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = "Optimization properties per optimization run",
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
    ax: matplotlib.axes.Axes | None = None,
    style_kwargs: dict | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot stats for all properties specified in properties_to_plot on one plot.

    Accepts one or more results. When multiple results are given, only a single
    property may be specified (one color per result). For a grid layout with one
    subplot per property see :func:`optimization_run_properties_subplots`.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py', or a list of those.
    ax:
        Axes object to use.
    properties_to_plot:
        Optimization run properties that should be plotted
    size:
        Figure size (width, height) in inches. Is only applied when no
        ``axes`` object is specified.
    title:
        Axes title. Pass ``None`` to suppress.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    colors:
        List of colors recognized by matplotlib colors (one color per property in properties_to_plot),
        or single color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels, one label per optimization property
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``ref_line_color`` — color of the connecting line in line-plot mode.
        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry
          (line-plot mode).
        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — histogram bar styling (hist-plot mode).

    Returns
    -------
    ax:
        The plot axes.

    Examples
    --------
    .. code-block:: python

        optimization_run_properties_one_plot(
            result1,
            properties_to_plot=['time'],
            colors=[.5, .9, .9, .3]
        )

        optimization_run_properties_one_plot(
            result1,
            properties_to_plot=['time', 'n_grad'],
            colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]]
        )
    """
    if properties_to_plot is None:
        properties_to_plot = [
            "time",
            "n_fval",
            "n_grad",
            "n_hess",
            "n_res",
            "n_sres",
        ]

    multi_result = isinstance(results, list) and len(results) > 1

    if multi_result and len(properties_to_plot) > 1:
        raise ValueError(
            "Passing multiple results and multiple properties at the same time is not "
            "supported on a single axis — the plot becomes too crowded to read. "
            "Either pass a single property to compare results, or a single result "
            "with multiple properties."
        )

    if plot_type != "line" and len(properties_to_plot) > 1:
        raise ValueError(
            f"plot_type='{plot_type}' with multiple properties does not make sense "
            "on a single axis (properties have different units/scales). "
            "Use plot_type='line' to overlay multiple properties, or use "
            "optimization_run_properties_subplots() for one subplot per property."
        )

    ax = get_ax(ax, size)

    style = resolve_style(style_kwargs)

    # Call stats_lowlevel directly (not via optimization_run_property_per_multistart)
    # to avoid the ax.clear() that the latter performs on every call.
    if multi_result:
        # Multiple results, single property: one color per result.
        result_list, colors_list, legends_list = process_result_list(
            results, colors, legends, style=style
        )
        prop_name = properties_to_plot[0]
        for result, color, legend in zip(result_list, colors_list, legends_list):
            stats_lowlevel(
                result, prop_name, prop_name, ax,
                start_indices, color, legend, plot_type, style=style,
            )
        # legends to inspect for the final ax.legend() call below
        legends = legends_list
    else:
        # Single result, possibly multiple properties: one color per property.
        single_result = results if not isinstance(results, list) else results[0]
        if colors is None:
            colors = assign_colors_for_list(
                len(properties_to_plot), style=style
            )
        elif is_color_like(colors):
            colors = [colors]
        if len(colors) != len(properties_to_plot):
            raise ValueError(
                "Number of colors should be the same as number of properties to plot."
            )
        if legends is None:
            legends = properties_to_plot
        elif not isinstance(legends, list):
            legends = [legends]
        if len(legends) != len(properties_to_plot):
            raise ValueError(
                "Number of legends should be the same as number of properties to plot."
            )
        for idx, (prop_name, legend) in enumerate(zip(properties_to_plot, legends)):
            stats_lowlevel(
                single_result, prop_name, prop_name, ax,
                start_indices, colors[idx], legend, plot_type, style=style,
            )

    _prop_labels = {
        "time": "Wall-clock time (s)", "n_fval": "Function evals",
        "n_grad": "Gradient evals", "n_hess": "Hessian evals",
        "n_res": "Residual evals", "n_sres": "Residual sens. evals",
    }
    if plot_type == "line":
        if len(properties_to_plot) == 1:
            ax.set_ylabel(_prop_labels.get(properties_to_plot[0], properties_to_plot[0]))
        else:
            ax.set_ylabel("Property value")
    if title is not None:
        ax.set_title(title)
    if any(leg is not None for leg in legends):
        ax.legend()
    return ax


def optimization_run_properties_subplots(
    results: Result | Sequence[Result],
    properties_to_plot: list[str] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
    axes: np.ndarray | None = None,
    style_kwargs: dict | None = None,
) -> np.ndarray:
    """
    One subplot per optimization property in a grid layout.

    The x-axis of every subplot shows individual optimizer runs ordered by
    objective value ("per multistart"). The difference from
    :func:`optimization_run_properties_one_plot` is layout only: this function
    gives each property its own panel with its own y-scale, while the other
    overlays all properties on a single axis.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    properties_to_plot:
        Optimization run properties that should be plotted
    size:
        Figure size (width, height) in inches. Is only applied when no
        ``axes`` object is specified.
    title:
        Figure title.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    colors:
        List of colors recognized by matplotlib (one color per result in results),
        or single color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``ref_line_color`` — color of the connecting line in line-plot mode.
        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry
          (line-plot mode).
        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — histogram bar styling (hist-plot mode).

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.

    Examples
    --------
    .. code-block:: python

        optimization_run_properties_subplots(
            result1,
            properties_to_plot=['time'],
            colors=[.5, .9, .9, .3]
        )

        optimization_run_properties_subplots(
            [result1, result2],
            properties_to_plot=['time'],
            colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]]
        )

        optimization_run_properties_subplots(
            result1,
            properties_to_plot=['time', 'n_grad'],
            colors=[.5, .9, .9, .3]
        )

        optimization_run_properties_subplots(
            [result1, result2], properties_to_plot=['time', 'n_fval'],
            colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]]
        )
    """
    if properties_to_plot is None:
        properties_to_plot = [
            "time",
            "n_fval",
            "n_grad",
            "n_hess",
            "n_res",
            "n_sres",
        ]

    if plot_type not in {"line", "hist"}:
        raise ValueError(
            "`optimization_run_properties_subplots` supports only "
            "`plot_type='line'` or `plot_type='hist'`."
        )

    num_subplot = len(properties_to_plot)
    num_row, num_col = make_grid_shape(num_subplot)
    axes = get_axes_array(axes=axes, nrows=num_row, ncols=num_col, size=size)
    axes = hide_unused_axes(axes=axes, n_used=num_subplot, clear=True)
    for idx, prop_name in enumerate(properties_to_plot):
        optimization_run_property_per_multistart(
            results,
            prop_name,
            axes=axes.flat[idx],
            size=size,
            start_indices=start_indices,
            colors=colors,
            legends=legends,
            plot_type=plot_type,
            style_kwargs=style_kwargs,
            show_legend=(idx == 0),
        )
    if title is not None:
        axes.flat[0].figure.suptitle(title)
    return axes


def optimization_run_property_per_multistart(
    results: Result | Sequence[Result],
    opt_run_property: str,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
    style_kwargs: dict | None = None,
    show_legend: bool = True,
) -> np.ndarray:
    """
    Plot stats for an optimization run property specified by opt_run_property.

    It is possible to plot a histogram or a line plot. In a line plot,
    on the x-axis are the numbers of the multistarts, where the multistarts are
    ordered with respect to a function value. On the y-axis of the line plot
    the value of the corresponding parameter for each multistart is displayed.

    Parameters
    ----------
    opt_run_property:
        optimization run property to plot.
        One of the 'time', 'n_fval', 'n_grad', 'n_hess', 'n_res', 'n_sres'
    results:
        Optimization result obtained by 'optimize.py' or list of those
    axes:
        Axes object or axes grid to use.
    size:
        Figure size (width, height) in inches. Is only applied when no
        ``axes`` object is specified.
    title:
        Figure title.
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    colors:
        List of colors recognized by matplotlib (one color per result in results),
        or single color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line', 'hist', 'both'
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``ref_line_color`` — color of the connecting line in line-plot mode.
        - ``scatter_size``, ``scatter_alpha``, ``scatter_linewidths``,
          ``scatter_edgecolors``, ``scatter_zorder`` — scatter point geometry
          (line-plot mode).
        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — histogram bar styling (hist/both-plot mode).
    show_legend:
        Whether to draw a legend on the axes. Set to ``False`` when this
        function is called for individual panels inside a grid (e.g. by
        :func:`optimization_run_properties_subplots`) so the legend appears
        only once.

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    supported_properties = {
        "time": "Wall-clock time (s)",
        "n_fval": "Function evals",
        "n_grad": "Gradient evals",
        "n_hess": "Hessian evals",
        "n_res": "Residual evals",
        "n_sres": "Residual sens. evals",
    }

    if opt_run_property not in supported_properties:
        raise ValueError(
            "Wrong value of opt_run_property. Only the following "
            "values are allowed: 'time', 'n_fval', 'n_grad', "
            "'n_hess', 'n_res', 'n_sres'"
        )

    style = resolve_style(style_kwargs)

    # parse input
    (results, colors, legends) = process_result_list(
        results, colors, legends, style=style
    )

    ncols = 2 if plot_type == "both" else 1
    axes = get_axes_array(axes=axes, nrows=1, ncols=ncols, size=size)
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)


    # loop over results
    for j, result in enumerate(results):
        if plot_type == "both":
            stats_lowlevel(
                result,
                opt_run_property,
                supported_properties[opt_run_property],
                axes[0, 0],
                start_indices,
                colors[j],
                legends[j],
                style=style,
            )

            stats_lowlevel(
                result,
                opt_run_property,
                supported_properties[opt_run_property],
                axes[0, 1],
                start_indices,
                colors[j],
                legends[j],
                plot_type="hist",
                style=style,
            )
        else:
            stats_lowlevel(
                result,
                opt_run_property,
                supported_properties[opt_run_property],
                axes[0, 0],
                start_indices,
                colors[j],
                legends[j],
                plot_type,
                style=style,
            )

    if show_legend:
        if sum(legend is not None for legend in legends) > 0:
            # multi-result: one legend entry per result label
            axes[0, 0].legend()
        elif plot_type in ("line", "both") and len(results) == 1 and colors[0] is None:
            # single-result cluster legend — only show once (first/line panel)
            fvals = results[0].optimize_result.fval
            clust_colors = assign_colors(
                fvals, None, balance_alpha=False, style=style
            )
            clusters, cluster_size = assign_clusters(np.asarray(fvals))
            handles = cluster_legend_handles_from_data(
                clusters, cluster_size, np.array(clust_colors)
            )
            if handles:
                axes[0, 0].legend(handles=handles)

    if title is not None:
        axes.flat[0].figure.suptitle(title)

    return axes


def stats_lowlevel(
    result: Result,
    property_name: str,
    axis_label: str,
    ax: matplotlib.axes.Axes,
    start_indices: int | Iterable[int] | None = None,
    color: COLOR | list[COLOR] | np.ndarray | None = "C0",
    legend: str | None = None,
    plot_type: str = "line",
    style: dict | None = None,
):
    """
    Plot values of the optimization run property across different multistarts.

    Parameters
    ----------
    result:
        Optimization result obtained by 'optimize.py'
    property_name:
        name of the optimization result property which value should be plotted
    axis_label:
        Label for the y-axis of the line plot or x-axis of the histogram
    ax:
        Axes object to use
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    color:
        List of colors recognized by matplotlib (length equal to the number of multistarts),
        or single color
        If not set, then for the line plot clustering is done and
        colors are assigned automatically
    legend:
        Label describing the result
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'
    style:
        Resolved pyPESTO visualization style. Forwarded to
        :func:`assign_colors`. Defaults to the global style.

    Returns
    -------
    ax:
        The plot axes.
    """
    if style is None:
        style = resolve_style()

    fvals = result.optimize_result.fval
    values = [[res[property_name]] for res in result.optimize_result.list]
    values, fvals = delete_nan_inf(fvals, values)

    if start_indices is not None:
        start_indices = process_start_indices(result, start_indices)
        values = values[start_indices]
        fvals = fvals[start_indices]

    n_starts = len(values)

    # assign colors
    colors = assign_colors(
        vals=fvals, colors=color, balance_alpha=False, style=style
    )

    sorted_indices = sorted(range(n_starts), key=lambda j: fvals[j])
    values = values[sorted_indices]

    if plot_type == "line":
        # plot line
        ax.plot(range(n_starts), values, color=style["ref_line_color"])

        # plot points
        for i, v in enumerate(values):
            if i == 0:
                tmp_legend = legend
            else:
                tmp_legend = None
            ax.scatter(
                i, v,
                color=colors[i],
                marker="o",
                s=style["scatter_size"],
                alpha=style["scatter_alpha"],
                linewidths=style["scatter_linewidths"],
                edgecolors=style["scatter_edgecolors"],
                zorder=style["scatter_zorder"],
                label=tmp_legend,
            )
        ax.set_xlabel("Ordered optimizer run")
        ax.set_ylabel(axis_label)
    else:
        ax.hist(
            values,
            color=style["rectangle_color"] if color is None else color,
            bins="auto",
            alpha=style["rectangle_alpha"],
            edgecolor=style["rectangle_edgecolor"],
            linewidth=style["rectangle_linewidth"],
            label=legend,
        )
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Multistarts")

    return ax
