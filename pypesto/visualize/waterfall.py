import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .reference_points import create_references
from .clust_color import assign_clustered_colors


def waterfall(result,
              ax=None,
              size=(18.5, 10.5),
              y_limits=None,
              start_indices=None,
              reference=None):
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

    start_indices: list or int
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # extract specific cost function values from result
    fvals = get_fvals(result, start_indices)

    # parse and apply plotting options
    ref = create_references(references=reference)

    # call lowlevel plot routine
    ax = waterfall_lowlevel(fvals, ax, size)

    # handle options
    ax = handle_options(ax, fvals, ref, y_limits)

    return ax


def waterfall_lowlevel(fvals, ax=None, size=(18.5, 10.5)):
    """
    Plot waterfall plot using list of function values.

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

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
    colors = assign_clustered_colors(fvals)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.plot(start_ind, fvals)
    for j in range(n_fvals):
        j_fval = indices[j]
        color = colors[j_fval]
        fval = fvals[j_fval]
        ax.plot(j, fval, color=color, marker='o')

    # labels
    ax.set_xlabel('Ordered optimizer run')
    ax.set_ylabel('Function value')
    ax.set_title('Waterfall plot')

    return ax


def get_fvals(result, start_indices):
    """
    Get function values from results.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

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

    # get only the indices which the user asked for
    return fvals[start_indices]


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
    if len(ref) > 0:
        # create set of colors for reference points
        ref_len = len(ref)
        for i_num, i_ref in enumerate(ref):
            ax.plot([0, len(fvals) - 1],
                    [i_ref.fval, i_ref.fval],
                    '--',
                    color=[0., 0.5 * (1. + i_num / ref_len), 0., 0.9])

    return ax
