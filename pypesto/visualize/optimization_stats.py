from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from .clust_color import assign_colors, delete_nan_inf, assign_colors_for_list
from ..result import Result
from .misc import process_result_list, process_start_indices


def optimization_run_properties_one_plot(
    results: Result,
    properties_to_plot: Optional[List[str]] = None,
    size: Tuple[float, float] = (18.5, 10.5),
    start_indices: Optional[Union[int, Iterable[int]]] = None,
    colors: Optional[Union[List[float], List[List[float]]]] = None,
    legends: Optional[Union[str, List[str]]] = None,
    plot_type: str = 'line'
) -> matplotlib.axes.Axes:
    """
    Plot stats for all optimization properties specified  in properties_to_plot
    on one plot.

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
        List of RGBA colors (one color per property in properties_to_plot),
        or single RGBA color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels, one label per optimization property
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'

    Returns
    -------

    Examples
    -------
    optimization_properties_per_multistart(
        result1,
        properties_to_plot=['time'],
        colors=[.5, .9, .9, .3])

    optimization_properties_per_multistart(
        result1,
        properties_to_plot=['time', 'n_grad'],
        colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]])
    """
    if properties_to_plot is None:
        properties_to_plot = ['time', 'n_fval', 'n_grad', 'n_hess', 'n_res',
                              'n_sres']

    if colors is None:
        colors = assign_colors_for_list(len(properties_to_plot))
    elif len(colors) == 4 and isinstance(colors[0], Real):
        colors = [colors]

    if len(colors) != len(properties_to_plot):
        raise ValueError('Number of RGBA colors should be the same as number '
                         'of optimization properties to plot')

    if legends is None:
        legends = properties_to_plot
    elif not isinstance(legends, list):
        legends = [legends]

    if len(legends) != len(properties_to_plot):
        raise ValueError('Number of legends should be the same as number of '
                         'optimization properties to plot')

    ax = plt.subplots()[1]
    fig = plt.gcf()
    fig.set_size_inches(*size)

    for idx, prop_name in enumerate(properties_to_plot):
        optimization_run_property_per_multistart(
            results, prop_name, ax, size, start_indices, colors[idx],
            legends[idx], plot_type)

    ax.set_ylabel("property value")
    ax.set_title("Optimization properties per optimization run")
    return ax


def optimization_run_properties_per_multistart(
        results: Union[Result, Sequence[Result]],
        properties_to_plot: Optional[List[str]] = None,
        size: Tuple[float, float] = (18.5, 10.5),
        start_indices: Optional[Union[int, Iterable[int]]] = None,
        colors: Optional[Union[List[float], List[List[float]]]] = None,
        legends: Optional[Union[str, List[str]]] = None,
        plot_type: str = 'line'
) -> Dict[str, plt.Subplot]:
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
        List of RGBA colors (one color per result in results),
        or single RGBA color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'

    Returns
    -------
    ax:
    The plot axes.


    Examples
    -------
    optimization_properties_per_multistart(
        result1,
        properties_to_plot=['time'],
        colors=[.5, .9, .9, .3])

    optimization_properties_per_multistart(
        [result1, result2],
        properties_to_plot=['time'],
        colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]])

    optimization_properties_per_multistart(
        result1,
        properties_to_plot=['time', 'n_grad'],
        colors=[.5, .9, .9, .3])

    optimization_properties_per_multistart(
        [result1, result2], properties_to_plot=['time', 'n_fval'],
        colors=[[.5, .9, .9, .3], [.2, .1, .9, .5]])
    """

    if properties_to_plot is None:
        properties_to_plot = ['time', 'n_fval', 'n_grad', 'n_hess', 'n_res',
                              'n_sres']

    num_subplot = len(properties_to_plot)
    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(num_subplot)))
    num_col = int(np.ceil(num_subplot / num_row))
    fig, axes = plt.subplots(num_row, num_col, squeeze=False)
    fig.set_size_inches(*size)

    for ax in axes.flat[num_subplot:]:
        ax.remove()
    axes = dict(zip(range(num_subplot), axes.flat))
    for idx, prop_name in enumerate(properties_to_plot):
        ax = axes[idx]
        optimization_run_property_per_multistart(
            results, prop_name, ax, size, start_indices, colors, legends,
            plot_type)
    return axes


def optimization_run_property_per_multistart(
        results: Union[Result, Sequence[Result]],
        opt_run_property: str,
        axes: Optional[matplotlib.axes.Axes] = None,
        size: Tuple[float, float] = (18.5, 10.5),
        start_indices: Optional[Union[int, Iterable[int]]] = None,
        colors: Optional[Union[List[float], List[List[float]]]] = None,
        legends: Optional[Union[str, List[str]]] = None,
        plot_type: str = 'line') -> matplotlib.axes.Axes:
    """
    Plot stats for an optimization run property specified by opt_run_property.
    It is possible to plot a histogram or a line plot. In a line plot,
    on the x axis are the numbers of the multistarts, where the multistarts are
    ordered with respect to a function value. On the y axis of the line plot
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
        List of RGBA colors (one color per result in results),
        or single RGBA color. If not set and one result, clustering is done
        and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line', 'hist', 'both'

    Returns
    -------
    ax:
        The plot axes.
    """

    supported_properties = {
        'time': 'Wall-clock time (seconds)',
        'n_fval': 'Number of function evaluations',
        'n_grad': 'Number of gradient evaluations',
        'n_hess': 'Number of Hessian evaluations',
        'n_res': 'Number of residuals evaluations',
        'n_sres': 'Number of residual sensitivity evaluations'
    }

    if opt_run_property not in supported_properties:
        raise ValueError("Wrong value of opt_run_property. Only the following "
                         "values are allowed: 'time', 'n_fval', 'n_grad', "
                         "'n_hess', 'n_res', 'n_sres'")

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    # axes
    if axes is None:
        ncols = 2 if plot_type == 'both' else 1
        fig, axes = plt.subplots(1, ncols)
        fig.set_size_inches(*size)
        fig.suptitle(
            f'{supported_properties[opt_run_property]} per optimizer run')
    else:
        axes.set_title(
            f'{supported_properties[opt_run_property]} per optimizer run')

    # loop over results
    for j, result in enumerate(results):
        if plot_type == 'both':
            axes[0] = stats_lowlevel(result, opt_run_property,
                                     supported_properties[opt_run_property],
                                     axes[0], start_indices, colors[j],
                                     legends[j])

            axes[1] = stats_lowlevel(result, opt_run_property,
                                     supported_properties[opt_run_property],
                                     axes[1], start_indices, colors[j],
                                     legends[j], plot_type='hist')
        else:
            axes = stats_lowlevel(result, opt_run_property,
                                  supported_properties[opt_run_property], axes,
                                  start_indices, colors[j], legends[j],
                                  plot_type)

    if sum((legend is not None for legend in legends)) > 0:
        if plot_type == 'both':
            for ax in axes:
                ax.legend()
        else:
            axes.legend()

    return axes


def stats_lowlevel(result: Result,
                   property_name: str,
                   axis_label: str,
                   ax: matplotlib.axes.Axes,
                   start_indices: Optional[Union[int, Iterable[int]]] = None,
                   color: Union[str, List[float], List[List[float]]] = 'C0',
                   legend: Optional[str] = None,
                   plot_type: str = 'line'):
    """
    Plot values of the optimization run property specified by property name
    across different multistarts

    Parameters
    ----------

    result:
        Optimization result obtained by 'optimize.py'
    property_name:
        name of the optimization result property which value should be plotted
    axis_label:
        Label for the y axis of the line plot or x axis of the histogram
    ax:
        Axes object to use
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    color:
        List of RGBA colors (length equal to the number of multistarts),
        or single color, defined by a string or RGBA list
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

    fvals = result.optimize_result.get_for_key('fval')
    values = result.optimize_result.get_for_key(property_name)
    values, fvals = delete_nan_inf(fvals, values)

    if start_indices is not None:
        start_indices = process_start_indices(start_indices, len(values))
        values = values[start_indices]
        fvals = fvals[start_indices]

    n_starts = len(values)

    # assign colors
    colors = assign_colors(vals=fvals, colors=color,
                           balance_alpha=False)

    # sort TODO: issue # 378
    sorted_indices = sorted(range(n_starts), key=lambda j: fvals[j])
    values = values[sorted_indices]

    if plot_type == 'line':
        # plot line
        ax.plot(range(n_starts), values, color=[0.7, 0.7, 0.7, 0.6])

        # plot points
        for i, v in enumerate(values):
            if i == 0:
                tmp_legend = legend
            else:
                tmp_legend = None
            ax.scatter(i, v, color=colors[i], marker='o', label=tmp_legend)
        ax.set_xlabel('Ordered optimizer run')
        ax.set_ylabel(axis_label)
    else:
        ax.hist(values, color=color, bins='auto', label=legend)
        ax.set_xlabel(axis_label)
        ax.set_ylabel('Number of multistarts')

    return ax
