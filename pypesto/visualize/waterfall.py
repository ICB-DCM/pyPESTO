import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .reference_points import create_references, ReferencePoint
from .clust_color import assign_colors, delete_nan_inf, RGBA
from .misc import (
    process_result_list, process_start_indices, process_y_limits,
    process_offset_y
)

from pypesto import Result
from typing import Optional, Union, Sequence, Tuple, List


def waterfall(results: Union[Result, Sequence[Result]],
              ax: Optional[plt.Axes] = None,
              size: Optional[Tuple[float]] = (18.5, 10.5),
              y_limits: Optional[Tuple[float]] = None,
              scale_y: Optional[str] = 'log10',
              offset_y: Optional[float] = None,
              start_indices: Optional[Union[Sequence[int], int]] = None,
              reference: Optional[Sequence[ReferencePoint]] = None,
              colors: Optional[Union[RGBA, Sequence[RGBA]]] = None,
              legends: Optional[Union[Sequence[str], str]] = None):
    """
    Plot waterfall plot.

    Parameters
    ----------

    results:
        Optimization result obtained by 'optimize.py' or list of those

    ax: matplotlib.Axes, optional
        Axes object to use.

    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits

    scale_y:
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, if it is supposed to be in log10-scale

    start_indices:
        Integers specifying the multistart to be plotted or int specifying
        up to which start index should be plotted

    reference:
        Reference points for optimization results, containing at least a
        function value fval

    colors:
        Colors or single color  for plotting. If not set, clustering is done
        and colors are assigned automatically

    legends:
        Labels for line plots, one label per result object

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    refs = create_references(references=reference)

    # precompute y-offset, if needed and if a list of results was passed
    fvals_all, offset_y = process_offset_for_list(offset_y, results, scale_y,
                                                  start_indices, refs)

    # plotting routine needs the maximum number of multistarts
    max_len_fvals = np.array([0])

    # loop over results
    for j, fvals in enumerate(fvals_all):
        # extract specific cost function values from result
        max_len_fvals = np.max([max_len_fvals, *fvals.shape])

        # call lowlevel plot routine
        ax = waterfall_lowlevel(fvals=fvals, scale_y=scale_y,
                                offset_y=offset_y, ax=ax, size=size,
                                colors=colors[j], legend_text=legends[j])

    # apply changes specified be the user to the axis object
    ax = handle_options(ax, max_len_fvals, refs, y_limits, offset_y)
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

    fvals.sort()
    n_fvals = len(fvals)
    start_ind = range(n_fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_colors(fvals, colors=colors)

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
        color = colors[j]
        fval = fvals[j]
        if j == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # line plot (linear or logarithmic)
        if scale_y == 'log10':
            ax.semilogy(j, fval, color=color,
                        marker='o', label=tmp_legend, alpha=1.)
        else:
            ax.plot(j, fval, color=color,
                    marker='o', label=tmp_legend, alpha=1.)

    # check if y-axis has a reasonable scale
    y_min, y_max = ax.get_ylim()
    if scale_y == 'log10':
        if np.log10(y_max) - np.log10(y_min) < 1.:
            y_mean = 0.5 * (np.log10(y_min) + np.log10(y_max))
            plt.ylim((10. ** (y_mean - 0.5), 10. ** (y_mean + 0.5)))
    else:
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


def process_offset_for_list(
        offset_y: float,
        results: Sequence[Result],
        scale_y: Optional[str],
        start_indices: Optional[Sequence[int]] = None,
        references: Optional[Sequence[ReferencePoint]] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    If we have a list of results, all should use the same offset_y,
    which is computed by this function and added to the fvals

    Parameters
    ----------

    offset_y:
        User provided offset_y

    results:
        Optimization results obtained by 'optimize.py'

    scale_y:
        May be logarithmic or linear ('log10' or 'lin')

    start_indices:
        Integers specifying the multistart to be plotted or int specifying
        up to which start index should be plotted


    references:
        Reference points that will be plotted along with the results

    Returns
    -------

    fvals:
        List of arrays of function values for each result

    offset_y:
        offset for the y-axis
    """
    min_val = np.inf
    fvals_all = []
    for result in results:
        fvals = np.asarray([
            np.array(result.optimize_result.get_for_key('fval'))
        ])
        if start_indices is None:
            start_indices = np.array(range(fvals.size))
        else:
            start_indices = process_start_indices(start_indices, fvals.size)
        fvals = fvals[:, start_indices]
        # if none of the fvals are finite, set default value to zero as
        # np.nanmin will error for an empty array
        if np.isfinite(fvals).any():
            min_val = min(min_val, np.nanmin(fvals[np.isfinite(fvals)]))

        fvals_all.append(fvals)

    # if there are references, also account for those
    if references:
        min_val = min(min_val, np.nanmin([r['fval'] for r in references]))

    offset_y = process_offset_y(offset_y, scale_y, float(min_val))

    # return offsetted values
    return [fvals + offset_y for fvals in fvals_all], offset_y


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

    # handle reference points
    for i_ref in ref:
        # plot reference point as line
        ax.plot([0, max_len_fvals - 1],
                [i_ref.fval + offset_y, i_ref.fval + offset_y], '--',
                color=i_ref.color, label=i_ref.legend)

        # create legend for reference points
        if i_ref.legend is not None:
            ax.legend()

    # handle y-limits
    ax = process_y_limits(ax, y_limits)

    return ax
