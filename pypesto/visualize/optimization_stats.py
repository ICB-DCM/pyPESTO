import matplotlib.axes
import matplotlib.pyplot as plt

from typing import Iterable, List, Optional, Sequence, Tuple, Union

from .clust_color import assign_colors, delete_nan_inf
from ..result import Result
from .misc import process_result_list, process_start_indices


def number_of_steps(
        results: Union[Result, Sequence[Result]],
        ax: Optional[matplotlib.axes.Axes] = None,
        size: Tuple[float, float] = (18.5, 10.5),
        start_indices: Optional[Union[int, Iterable[int]]] = None,
        colors: Optional[Union[List[float], List[List[float]]]] = None,
        legends: Optional[Union[str, List[str]]] = None,
        plot_type: str = 'line') -> matplotlib.axes.Axes:
    """
    Plot number of steps stats.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax:
        Axes object to use
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    colors:
        List of RGBA colors, or single RGBA color
        If not set, clustering is done and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'

    Returns
    -------
    ax:
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # loop over results
    for j, result in enumerate(results):
        ax = stats_lowlevel(result, 'n_fval', 'Number of steps', ax,
                            start_indices, colors[j], legends[j], plot_type)

    ax.set_title('Number of steps per optimizer run')
    if sum([l is not None for l in legends]) > 0:
        ax.legend()

    return ax


def optimization_time(
        results: Union[Result, Sequence[Result]],
        ax: Optional[matplotlib.axes.Axes] = None,
        size: Tuple[float, float] = (18.5, 10.5),
        start_indices: Optional[Union[int, Iterable[int]]] = None,
        colors: Optional[Union[List[float], List[List[float]]]] = None,
        legends: Optional[Union[str, List[str]]] = None,
        plot_type: str = 'line') -> matplotlib.axes.Axes:
    """
    Plot optimization duration stats.
    See pypesto.optimize.optimizer.time_decorator

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax:
        Axes object to use
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    colors:
        List of RGBA colors, or single RGBA color
        If not set, clustering is done and colors are assigned automatically
    legends:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'

    Returns
    -------
    ax:
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # loop over results
    for j, result in enumerate(results):
        ax = stats_lowlevel(result, 'time', 'Wall-clock time, seconds', ax,
                            start_indices, colors[j], legends[j], plot_type)

    ax.set_title('Wall-clock time per optimizer run')
    if sum([l is not None for l in legends]) > 0:
        ax.legend()

    return ax


def stats_lowlevel(result: Result,
                   key: str,
                   axis_label: str,
                   ax: Optional[matplotlib.axes.Axes] = None,
                   start_indices: Optional[Union[int, Iterable[int]]] = None,
                   color: Optional = 'C0',
                   legend: Optional[Union[str, List[str]]] = None,
                   plot_type: str = 'line'):
    """
    Plotting stats of the values defined by the key attribute across different
    multistars for an optimization result

    Parameters
    ----------

    result:
        Optimization result obtained by 'optimize.py'
    key:
        key of the optimization result which value should be plotted
    axis_label:
        Label for the y axis of the line plot or x axis of the histogram
    ax:
        Axes object to use
    start_indices:
        List of integers specifying the multistarts to be plotted or
        int specifying up to which start index should be plotted
    color:
        List of RGBA colors, or single RGBA color
        If not set, then for the line plot clustering is done and
        colors are assigned automatically
    legend:
        Labels for line plots, one label per result object
    plot_type:
        Specifies plot type. Possible values: 'line' and 'hist'

    Returns
    -------
    ax:
        The plot axes.
    """

    fvals = result.optimize_result.get_for_key('fval')
    values = result.optimize_result.get_for_key(key)
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
