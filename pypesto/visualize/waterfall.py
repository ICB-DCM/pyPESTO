import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings
from .reference_points import create_references
from .clust_color import assign_colors


def waterfall(result,
              ax=None,
              size=(18.5, 10.5),
              y_limits=None,
              scale_y='log10',
              offset_y=None,
              start_indices=None,
              reference=None,
              colors=None):
    """
    Plot waterfall plot.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

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
        offset for the y-axis, if this is supposed to be in log10-scale

    start_indices: list or int
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGB, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # extract specific cost function values from result
    fvals = get_fvals(result, scale_y, offset_y, start_indices)

    # parse and apply plotting options
    ref = create_references(references=reference)

    # call lowlevel plot routine
    ax = waterfall_lowlevel(fvals=fvals, scale_y=scale_y, ax=ax, size=size,
                            colors=colors)

    # handle options
    ax = handle_options(ax, fvals, ref, y_limits)

    return ax


def waterfall_lowlevel(fvals, scale_y='log10', ax=None, size=(18.5, 10.5),
                       colors=None):
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

    colors: list, or RGB, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

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

    n_fvals = len(fvals)
    start_ind = range(n_fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    if colors is None:
        colors = assign_colors(fvals)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # plot line
    if scale_y == 'log10':
        ax.semilogy(start_ind, fvals)
    else:
        ax.plot(start_ind, fvals)

    # plot points
    for j in range(n_fvals):
        j_fval = indices[j]
        color = colors[j_fval]
        fval = fvals[j_fval]
        if scale_y == 'log10':
            ax.semilogy(j, fval, color=color, marker='o')
        else:
            ax.plot(j, fval, color=color, marker='o')

    # labels
    ax.set_xlabel('Ordered optimizer run')
    ax.set_ylabel('Function value')
    ax.set_title('Waterfall plot')

    return ax


def get_fvals(result, scale_y, offset_y, start_indices):
    """
    Get function values from results.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, if this is supposed to be in log10-scale

    start_indices: list or int
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    fvals: ndarray
        function values
    """

    # extract cost function values from result
    fvals = np.array(result.optimize_result.get_for_key('fval'))

    # get list of indices
    if start_indices is None:
        start_indices = np.array(range(len(fvals)))
    else:
        # check whether list or maximum value
        start_indices = np.array(start_indices)
        if start_indices.size == 1:
            start_indices = np.array(range(start_indices))

        # check, whether index set is not too big
        existing_indices = np.array(range(len(fvals)))
        start_indices = np.intersect1d(start_indices, existing_indices)

    # reduce to indices for which the user asked
    fvals = fvals[start_indices]

    # get the minimal value shich should be plotted
    min_val = np.min(fvals)

    # check whether the offset specified by the user is sufficient
    if offset_y is not None:
        if (scale_y == 'log10') and (min_val + offset_y <= 0.):
            warnings.warn("Offset specified by user is insufficient. "
                          "Ignoring specified offset and using " +
                          str(np.abs(min_val) + 1.) + " instead.")
            offset_y = 1. - min_val
    else:
        # check whether scaling is lin or log10
        if scale_y == 'lin':
            offset_y = 0.
        else:
            offset_y = 1. - min_val

    # apply offset
    if offset_y != 0.:
        fvals += offset_y * np.ones(fvals.shape)

    # get only the indices which the user asked for
    return fvals


def handle_options(ax, fvals, ref, y_limits):
    """
    Handle reference points.

    Parameters
    ----------

    ax: matplotlib.Axes, optional
        Axes object to use.

    fvals: numeric list or array
        Including values need to be plotted.

    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # handle y-limits
    if y_limits is not None:
        y_limits = np.array(y_limits)
        if y_limits.size == 1:
            tmp_y_limits = ax.get_ylim()
            y_limits = [tmp_y_limits[0], y_limits]
        else:
            y_limits = [y_limits[0], y_limits[1]]

        # set limits
        ax.set_ylim(y_limits)

    # handle reference points
    for i_ref in ref:
        ax.plot([0, len(fvals) - 1], [i_ref.fval, i_ref.fval], '--',
                color=i_ref.color)

    return ax
