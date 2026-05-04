from collections.abc import Iterable, Sequence

import matplotlib.axes
import numpy as np
from matplotlib.colors import is_color_like

from pypesto.util import delete_nan_inf

from ..C import COLOR
from ..result import Result
from ._style import get_ax, get_axes_array
from .clust_color import assign_colors, assign_colors_for_list
from .misc import process_result_list, process_start_indices


def optimization_run_properties_one_plot(
    results: Result,
    properties_to_plot: list[str] | None = None,
    size: tuple[float, float] = (18.5, 10.5),
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """
    Plot stats for allproperties specified in properties_to_plot on one plot.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax:
        Axes object to use.
    properties_to_plot:
        Optimization run properties that should be plotted
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
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

    if colors is None:
        colors = assign_colors_for_list(len(properties_to_plot))
    elif is_color_like(colors):
        colors = [colors]

    if len(colors) != len(properties_to_plot):
        raise ValueError(
            "Number of colors should be the same as number "
            "of optimization properties to plot"
        )

    if legends is None:
        legends = properties_to_plot
    elif not isinstance(legends, list):
        legends = [legends]

    if len(legends) != len(properties_to_plot):
        raise ValueError(
            "Number of legends should be the same as number of "
            "optimization properties to plot"
        )

    ax = get_ax(ax, size)

    for idx, prop_name in enumerate(properties_to_plot):
        optimization_run_property_per_multistart(
            results,
            prop_name,
            ax,
            size,
            start_indices,
            colors[idx],
            legends[idx],
            plot_type,
        )

    ax.set_ylabel("property value")
    ax.set_title("Optimization properties per optimization run")
    return ax


def optimization_run_properties_per_multistart(
    results: Result | Sequence[Result],
    properties_to_plot: list[str] | None = None,
    size: tuple[float, float] = (18.5, 10.5),
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """
    One plot per optimization property in properties_to_plot.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    properties_to_plot:
        Optimization run properties that should be plotted
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
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

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.

    Examples
    --------
    .. code-block:: python

        optimization_run_properties_per_multistart(
            result1,
            properties_to_plot=['time'],
            colors=[.5, .9, .9, .3]
        )

        optimization_run_properties_per_multistart(
            [result1, result2],
            properties_to_plot=['time'],
            colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]]
        )

        optimization_run_properties_per_multistart(
            result1,
            properties_to_plot=['time', 'n_grad'],
            colors=[.5, .9, .9, .3]
        )

        optimization_run_properties_per_multistart(
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
            "`optimization_run_properties_per_multistart` supports only "
            "`plot_type='line'` or `plot_type='hist'`."
        )

    num_subplot = len(properties_to_plot)
    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(num_subplot)))
    num_col = int(np.ceil(num_subplot / num_row))
    axes = get_axes_array(axes=axes, nrows=num_row, ncols=num_col, size=size)
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)
    for ax in axes.flat[num_subplot:]:
        ax.set_visible(False)
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
        )
    return axes


def optimization_run_property_per_multistart(
    results: Result | Sequence[Result],
    opt_run_property: str,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    size: tuple[float, float] = (18.5, 10.5),
    start_indices: int | Iterable[int] | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    legends: str | list[str] | None = None,
    plot_type: str = "line",
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
        Axes object to use
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
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

    Returns
    -------
    axes:
        2-D NumPy array containing one matplotlib Axes per panel.
    """
    supported_properties = {
        "time": "Wall-clock time (seconds)",
        "n_fval": "Number of function evaluations",
        "n_grad": "Number of gradient evaluations",
        "n_hess": "Number of Hessian evaluations",
        "n_res": "Number of residuals evaluations",
        "n_sres": "Number of residual sensitivity evaluations",
    }

    if opt_run_property not in supported_properties:
        raise ValueError(
            "Wrong value of opt_run_property. Only the following "
            "values are allowed: 'time', 'n_fval', 'n_grad', "
            "'n_hess', 'n_res', 'n_sres'"
        )

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    ncols = 2 if plot_type == "both" else 1
    axes = get_axes_array(axes=axes, nrows=1, ncols=ncols, size=size)
    fig = axes.flat[0].figure
    for ax in axes.flat:
        ax.clear()
        ax.set_visible(True)

    if plot_type == "both":
        fig.suptitle(
            f"{supported_properties[opt_run_property]} per optimizer run"
        )
    else:
        axes[0, 0].set_title(
            f"{supported_properties[opt_run_property]} per optimizer run"
        )

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
            )

    if sum(legend is not None for legend in legends) > 0:
        if plot_type == "both":
            for ax in axes.flat:
                ax.legend()
        else:
            axes[0, 0].legend()

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

    Returns
    -------
    ax:
        The plot axes.
    """
    fvals = result.optimize_result.fval
    values = [[res[property_name]] for res in result.optimize_result.list]
    values, fvals = delete_nan_inf(fvals, values)

    if start_indices is not None:
        start_indices = process_start_indices(result, start_indices)
        values = values[start_indices]
        fvals = fvals[start_indices]

    n_starts = len(values)

    # assign colors
    colors = assign_colors(vals=fvals, colors=color, balance_alpha=False)

    sorted_indices = sorted(range(n_starts), key=lambda j: fvals[j])
    values = values[sorted_indices]

    if plot_type == "line":
        # plot line
        ax.plot(range(n_starts), values, color=[0.7, 0.7, 0.7, 0.6])

        # plot points
        for i, v in enumerate(values):
            if i == 0:
                tmp_legend = legend
            else:
                tmp_legend = None
            ax.scatter(i, v, color=colors[i], marker="o", label=tmp_legend)
        ax.set_xlabel("Ordered optimizer run")
        ax.set_ylabel(axis_label)
    else:
        ax.hist(values, color=color, bins="auto", label=legend)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("Number of multistarts")

    return ax
