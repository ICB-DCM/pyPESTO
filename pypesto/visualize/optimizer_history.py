import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .visualization import handle_options
from .clust_color import assign_clustered_colors


def optimizer_history(result, ax=None,
                      size=(18.5, 10.5),
                      trace_x='steps',
                      trace_y='fval',
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

    # call lowlevel plot routine
    ax = optimizer_history_lowlevel(vals, ax, size)

    # set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Optimizer history')

    # parse and apply plotting options
    ref = handle_options(ax, None, reference)

    # plot reference points
    # get length of longest trajectory
    max_len = 0
    for val in vals:
        max_len = np.max([max_len, val[0, -1]])

    if ref is not None:
        ref_len = len(ref)
        for i_num, i_ref in enumerate(ref):
            ax.semilogy([0, max_len], [i_ref.fval, i_ref.fval], '--',
                        color=[0., 0.5 * (1. + i_num / ref_len), 0., 0.9])

    return ax


def optimizer_history_lowlevel(vals, ax=None, size=(18.5, 10.5)):
    """
    Plot optimizer history using list of numpy array.

    Parameters
    ----------

    vals:  list of numpy arrays
        list of 2xn-arrays (x_values and y_values of the trace)

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
        for val in vals:
            val = np.array(val)
            fvals.append(val[1, -1])
    n_fvals = len(fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_clustered_colors(fvals)

    # sort
    indices = sorted(range(n_fvals),
                     key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j, val in enumerate(vals):
        j_fval = indices[j]
        color = colors[j_fval]
        ax.semilogy(val[0, :], val[1, :], color=color)

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
            raise ('gradient norm history is not implemented yet!')
            y_label = 'Gradient norm'
        else:  # trace_y == 'fval':
            y_label = 'Objective value'
            y_vals = np.array(trace['fval'][indices])

        # write down values
        vals.append(np.array([x_vals, y_vals]))

    return x_label, y_label, vals
