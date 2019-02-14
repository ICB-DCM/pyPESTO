import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings
from .reference_points import create_references
from .clust_color import assign_colors


def optimizer_history(result, ax=None,
                      size=(18.5, 10.5),
                      trace_x='steps',
                      trace_y='fval',
                      scale_y='log10',
                      offset_y=None,
                      colors=None,
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

    trace_x: str, optional
        What should be plotted on the x-axis?
        Possibilities: 'time', 'steps'
        Default: 'steps'

    trace_y: str, optional
        What should be plotted on the y-axis?
        Possibilities: 'fval', 'gradnorm', 'stepsize'
        Default: 'fval'

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y: float, optional
        Offset for the y-axis-values, as these are plotted on a log10-scale
        Will be computed automatically if necessary

    colors: list, or RGB, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

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

    # extract cost function values from result
    (x_label, y_label, vals) = get_trace(result, trace_x, trace_y)

    # parse and apply plotting options
    ref = create_references(references=reference)

    # compute the necessary offset for the y-axis
    vals = get_vals(vals, scale_y, offset_y, start_indices)

    # call lowlevel plot routine
    ax = optimizer_history_lowlevel(vals, scale_y=scale_y, colors=colors,
                                    ax=ax, size=size)

    # handle options
    ax = handle_options(ax, vals, ref, y_limits, x_label, y_label)

    return ax


def optimizer_history_lowlevel(vals, scale_y='log10', colors=None, ax=None,
                               size=(18.5, 10.5)):
    """
    Plot optimizer history using list of numpy array.

    Parameters
    ----------

    vals:  list of numpy arrays
        list of 2xn-arrays (x_values and y_values of the trace)

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    colors: list, or RGB, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

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
    fvals = []
    if isinstance(vals, list):
        # convert entries to numpy arrays
        for val in vals:
            val = np.array(val)
            fvals.append(val[1, -1])
    else:
        # convert to a list of numpy arrays
        vals = np.array(vals)
        if vals.shape[0] != 2 or vals.ndim != 2:
            raise('If numpy array is passed directly to lowlevel routine of'
                  'optimizer_history, shape needs to be 2 x n.')
        fvals = [vals[1, -1]]
        vals = [vals]
    n_fvals = len(fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_colors(fvals, colors)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j, val in enumerate(vals):
        j_fval = indices[j]
        color = colors[j_fval]
        if scale_y == 'log10':
            ax.semilogy(val[0, :], val[1, :], color=color)
        else:
            ax.plot(val[0, :], val[1, :], color=color)

    return ax


def get_trace(result, trace_x, trace_y):
    """
    Handle bounds and results.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.

    trace_x: str, optional
        What should be plotted on the x-axis?
        Possibilities: 'time', 'steps'
        Default: 'steps'

    trace_y: str, optional
        What should be plotted on the y-axis?
        Possibilities: 'fval'(later also: 'gradnorm', 'stepsize')
        Default: 'fval'

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # get data frames
    traces = result.optimize_result.get_for_key('trace')
    vals = []

    for trace in traces:
        # SciPy optimizers seem to save more steps than actually taken: prune
        indices = np.argwhere(np.isfinite(trace['fval']))
        indices = indices.flatten()
        indices.astype(int)

        # retrieve values from dataframe
        if trace_x == 'time':
            x_vals = np.array(trace['time'][indices])
            x_label = 'Computation time [s]'

        else:  # trace_x == 'steps':
            x_vals = np.array(list(range(len(indices))))
            x_label = 'Optimizer steps'

        if trace_y == 'gradnorm':
            # retrieve gradient trace, if saved
            if trace['grad'] is None:
                raise("No gradient norm trace can be visualized: "
                      "The pypesto.result object does not contain "
                      "a gradient trace")

            # Get gradient trace, prune Nones, compute norm
            tmp_grad_trace = list(trace['grad'].values)
            y_vals = np.array([np.linalg.norm(grad) for grad in
                               tmp_grad_trace if grad is not None])
            y_label = 'Gradient norm'

        else:  # trace_y == 'fval':
            y_label = 'Objective value'
            y_vals = np.array(trace['fval'][indices])

        # write down values
        vals.append(np.array([x_vals, y_vals]))

    return x_label, y_label, vals


def get_vals(vals, scale_y, offset_y, start_indices):
    """
    Handle bounds and results.

    Parameters
    ----------

    vals: list
        list of 2xn-numpy arrays

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y:
        offset for the y-axis, as this is supposed to be in log10-scale

    start_indices: list or int
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted

    Returns
    -------

    vals: list
        list of 2xn-numpy arrays

    offset_y:
        offset for the y-axis, as this is supposed to be in log10-scale
    """

    # get list of indices
    if start_indices is None:
        start_indices = np.array(range(len(vals)))
    else:
        # check whether list or maximum value
        start_indices = np.array(start_indices)
        if start_indices.size == 1:
            start_indices = np.array(range(start_indices))

        # check, whether index set is not too big
        existing_indices = np.array(range(len(vals)))
        start_indices = np.intersect1d(start_indices, existing_indices)

    # reduce values to listed values
    vals = [val for i, val in enumerate(vals) if i in start_indices]

    # get the minimal value shich should be plotted
    min_val = np.inf
    for val in vals:
        tmp_min = np.min(val[1, :])
        min_val = np.min([min_val, tmp_min])

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
            offset_y = 0
        else:
            offset_y = 1. - min_val

    if offset_y != 0:
        for val in vals:
            val[1, :] += offset_y * np.ones(val[1].shape)

    return vals


def handle_options(ax, vals, ref, y_limits, x_label, y_label):
    """
    Handle reference points.

    Parameters
    ----------

    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    vals: list
        list of 2xn-numpy arrays

    ax: matplotlib.Axes, optional
        Axes object to use.

    y_limits: float or ndarray, optional
        maximum value to be plotted on the y-axis, or y-limits

    x_label: str
        label for x-axis

    y_label: str
        label for x-axis

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
    ax.set_ylim(y_limits)

    # handle reference points
    if len(ref) > 0:
        # plot reference points
        # get length of longest trajectory
        max_len = 0
        for val in vals:
            max_len = np.max([max_len, val[0, -1]])

        for i_ref in ref:
            ax.semilogy([0, max_len], [i_ref.fval, i_ref.fval], '--',
                        color=i_ref.color)

    # set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Optimizer history')

    return ax
