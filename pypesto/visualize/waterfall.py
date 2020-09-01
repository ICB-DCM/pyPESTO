import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .reference_points import create_references
from .clust_color import assign_colors
from .clust_color import delete_nan_inf
from .misc import process_result_list
from .misc import process_y_limits
from .misc import process_offset_y

from pypesto import Result
from typing import Iterable, Optional


def waterfall(results,
              ax=None,
              size=(18.5, 10.5),
              y_limits=None,
              scale_y='log10',
              offset_y=None,
              start_indices=None,
              reference=None,
              colors=None,
              legends=None):
    """
    Plot waterfall plot.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    start_indices: list or int
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    legends: list or str
        Labels for line plots, one label per result object

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """
    if isinstance(start_indices, int):
        start_indices = list(range(start_indices))

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    # precompute y-offset, if needed and if a list of results was passed
    if offset_y is None and len(results) > 1 and scale_y == 'log10':
        offset_y = process_offset_for_list(results, scale_y)

    # plotting routine needs the maximum number of multistarts
    max_len_fvals = np.array([0])

    # loop over results
    for j, result in enumerate(results):
        # extract specific cost function values from result
        (fvals, offset_y) = get_fvals(result, scale_y, offset_y, start_indices)
        max_len_fvals = np.max([max_len_fvals, len(fvals)])

        # call lowlevel plot routine
        ax = waterfall_lowlevel(fvals=fvals, scale_y=scale_y,
                                offset_y=offset_y, ax=ax, size=size,
                                colors=colors[j], legend_text=legends[j])

    # parse and apply plotting options
    ref = create_references(references=reference)

    # apply changes specified be the user to the axis object
    ax = handle_options(ax, max_len_fvals, ref, y_limits, offset_y)

    return ax


def waterfall_lowlevel(fvals, scale_y='log10', offset_y=0., ax=None,
                       size=(18.5, 10.5), colors=None, legend_text=None):
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    legend_text: str
        Label for line plots

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # parse input
    fvals = np.array(fvals)
    # remove nan or inf values in fvals
    _, fvals = delete_nan_inf(fvals)

    n_fvals = len(fvals)
    start_ind = range(n_fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_colors(fvals, colors=colors)

    # sort
    indices = sorted(range(n_fvals), key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plot line
    if scale_y == 'log10':
        ax.semilogy(start_ind, fvals, color=[0.7, 0.7, 0.7, 0.6])
    else:
        ax.plot(start_ind, fvals, color=[0.7, 0.7, 0.7, 0.6])

    # plot points
    for j in range(n_fvals):
        # parse data for plotting
        j_fval = indices[j]
        color = colors[j_fval]
        fval = fvals[j_fval]
        if j == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # line plot (linear or logarithmic)
        y_min, y_max = ax.get_ylim()
        if scale_y == 'log10':
            ax.semilogy(j, fval, color=color,
                        marker='o', label=tmp_legend, alpha=1.)

            # check if y-axis has a reasonable scale
            if np.log10(y_max) - np.log10(y_min) < 1.:
                y_mean = 0.5 * (np.log10(y_min) + np.log10(y_max))
                plt.ylim((10.**(y_mean - 0.5), 10.**(y_mean + 0.5)))
        else:
            ax.plot(j, fval, color=color,
                    marker='o', label=tmp_legend, alpha=1.)

            # check if y-axis has a reasonable scale
            if y_max - y_min < 1.:
                y_mean = 0.5 * (y_min + y_max)
                plt.ylim((y_mean - 0.5, y_mean + 0.5))

    # labels
    ax.set_xlabel('Ordered optimizer run')
    if offset_y == .0:
        ax.set_ylabel('Function value')
    else:
        ax.set_ylabel('Offsetted function value (relative to best start)')
    ax.set_title('Waterfall plot')
    if legend_text is not None:
        ax.legend()

    return ax


def get_fvals(result: Result,
              scale_y: str,
              offset_y: float,
              start_indices: Optional[Iterable[int]] = None):
    """
    Get function values to be plotted later from results.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    start_indices:
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    fvals: ndarray
        function values

    offset_y:
        offset for the y-axis, if this is supposed to be in log10-scale
    """

    # extract cost function values from result
    fvals = np.array(result.optimize_result.get_for_key('fval'))

    # get list of indices
    if start_indices is None:
        start_indices = np.array(range(len(fvals)))
    else:
        # check whether list or maximum value
        start_indices = np.array(start_indices)

        # check, whether index set is not too big
        existing_indices = np.array(range(len(fvals)))
        start_indices = np.intersect1d(start_indices, existing_indices)

    # reduce to indices for which the user asked
    fvals = fvals[start_indices]

    # get the minimal value which should be plotted,
    # avoid the value being -inf or nan
    min_val = np.nanmin(fvals[fvals != -np.inf])

    # check, whether offset can be used with this data
    offset_y = process_offset_y(offset_y, scale_y, float(min_val))

    # apply offset
    if offset_y != 0.:
        fvals += offset_y * np.ones(fvals.shape)

    # get only the indices which the user asked for
    return fvals, offset_y


def process_offset_for_list(results: Iterable[Result],
                            scale_y: str) -> float:
    """
    If we have a list of results, all should use the same offset_y,
    which is computed by this function.

    Parameters
    ----------

    results: list of pypesto.Result
        list of Optimization results obtained by 'optimize.py'

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    Returns
    -------

    offset_y:
        offset for the y-axis
    """

    fvals = np.concatenate([
        np.array(result.optimize_result.get_for_key('fval'))
        for result in results
    ])
    min_val = np.nanmin(fvals[np.isfinite(fvals)])
    offset_y = process_offset_y(None, scale_y, float(min_val))

    return offset_y


def handle_options(ax, max_len_fvals, ref, y_limits, offset_y):
    """
    Get the limits for the y-axis, plots the reference points, will do
    more at a later time point. This function is there to apply whatever
    kind of post-plotting transformations to the axis object.

    Parameters
    ----------

    ax: matplotlib.Axes, optional
        Axes object to use.

    max_len_fvals: int
        maximum number of points

    ref: list, optional
        List of reference points for optimization results, containing at
        least a function value fval

    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits

    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # handle y-limits
    ax = process_y_limits(ax, y_limits)

    # handle reference points
    for i_ref in ref:
        # plot reference point as line
        ax.plot([0, max_len_fvals - 1],
                [i_ref.fval + offset_y, i_ref.fval + offset_y], '--',
                color=i_ref.color, label=i_ref.legend)

        # create legend for reference points
        if i_ref.legend is not None:
            ax.legend()

    return ax
