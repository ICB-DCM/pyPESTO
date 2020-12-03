import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from typing import List, Optional, Tuple, Union, Iterable

from ..result import Result
from ..objective import History
from .reference_points import create_references, ReferencePoint
from .clust_color import assign_colors
from .misc import process_result_list
from .misc import process_y_limits
from .misc import process_offset_y


def optimizer_history(results,
                      ax=None,
                      size=(18.5, 10.5),
                      trace_x='steps',
                      trace_y='fval',
                      scale_y='log10',
                      offset_y=None,
                      colors=None,
                      y_limits=None,
                      start_indices=None,
                      reference=None,
                      legends=None):
    """
    Plot history of optimizer. Can plot either the history of the cost
    function or of the gradient norm, over either the optimizer steps or
    the computation time.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

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

    colors: list, or RGBA, optional
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

    for j, result in enumerate(results):
        # extract cost function values from result
        (x_label, y_label, vals) = get_trace(result, trace_x, trace_y)

        # compute the necessary offset for the y-axis
        (vals, offset_y, y_label) = get_vals(vals, scale_y, offset_y, y_label,
                                             start_indices)

        # call lowlevel plot routine
        ax = optimizer_history_lowlevel(vals, scale_y=scale_y, ax=ax,
                                        colors=colors[j],
                                        size=size, x_label=x_label,
                                        y_label=y_label,
                                        legend_text=legends[j])

    # parse and apply plotting options
    ref = create_references(references=reference)

    # handle options
    ax = handle_options(ax, vals, ref, y_limits, offset_y)

    return ax


def optimizer_history_lowlevel(vals, scale_y='log10', colors=None, ax=None,
                               size=(18.5, 10.5), x_label='Optimizer steps',
                               y_label='Objective value', legend_text=None):
    """
    Plot optimizer history using list of numpy arrays.

    Parameters
    ----------

    vals: list of numpy arrays
        list of 2xn-arrays (x_values and y_values of the trace)

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see waterfall

    x_label: str
        label for x-axis

    y_label: str
        label for y-axis

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
            raise ValueError('If numpy array is passed directly to lowlevel '
                             'routine of optimizer_history, shape needs to '
                             'be 2 x n.')
        fvals = [vals[1, -1]]
        vals = [vals]
    n_fvals = len(fvals)

    # assign colors
    # note: this has to happen before sorting
    # to get the same colors in different plots
    colors = assign_colors(fvals, colors)

    # sort
    indices = sorted(range(n_fvals), key=lambda j: fvals[j])

    # plot
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j, val in enumerate(vals):
        # collect and parse data
        j_fval = indices[j]
        color = colors[j_fval]
        if j == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # line plots
        if scale_y == 'log10':
            ax.semilogy(val[0, :], val[1, :], color=color, label=tmp_legend)
        else:
            ax.plot(val[0, :], val[1, :], color=color, label=tmp_legend)

    # set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Optimizer history')
    if legend_text is not None:
        ax.legend()

    return ax


def get_trace(result: Result,
              trace_x: Optional[str],
              trace_y: Optional[str]) -> Tuple[str, str, List[np.ndarray]]:
    """
    Get the values of the optimizer trace from the pypesto.Result object

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

    vals:
        list of

    x_label:
        label for x-axis to be plotted later

    y_label:
        label for y-axis to be plotted later
    """

    # get data frames
    histories: List[History] = result.optimize_result.get_for_key('history')

    vals = []
    x_label = ''
    y_label = ''

    for history in histories:
        options = history.options
        if trace_y == 'gradnorm':
            # retrieve gradient trace, if saved
            if not options.trace_record or not \
                    options.trace_record_grad:
                raise ValueError("No gradient trace has been recorded.")
            grads = history.get_grad_trace()
            indices = [i for i, val in enumerate(grads)
                       if val is not None and np.isfinite(val).all()]

            grads = np.array([grads[i] for i in indices])

            # Get gradient trace, compute norm
            y_vals = np.linalg.norm(grads, axis=1)
            y_label = 'gradient norm'

        else:  # trace_y == 'fval':
            if not options.trace_record:
                raise ValueError("No function value trace has been recorded.")
            fvals = history.get_fval_trace()
            indices = [i for i, val in enumerate(fvals)
                       if val is not None and np.isfinite(val)]

            y_vals = np.array([fvals[i] for i in indices])
            y_label = 'objective value'

        # retrieve values from dataframe
        if trace_x == 'time':
            times = np.array(history.get_time_trace())
            x_vals = times[indices]
            x_label = 'Computation time [s]'

        else:  # trace_x == 'steps':
            x_vals = np.array(list(range(len(indices))))
            x_label = 'Optimizer steps'

        # write down values
        vals.append(np.vstack([x_vals, y_vals]))

    return x_label, y_label, vals


def get_vals(
    vals: List[np.ndarray],
    scale_y: Optional[str],
    offset_y: float,
    y_label: str,
    start_indices: Iterable[int]
) -> Tuple[List[np.ndarray], float, str]:
    """
    Postprocesses the values of the optimization history, depending on the
    options set by the user (e.g. scale_y, offset_y, start_indices)

    Parameters
    ----------

    vals: list
        list of numpy arrays of dimension 2 x len(start_indices)

    scale_y: str, optional
        May be logarithmic or linear ('log10' or 'lin')

    offset_y: float
        offset for the y-axis, as this is supposed to be in log10-scale

    y_label: str
        Label for y axis

    start_indices:
        list of integers specifying the multi start indices to be plotted

    Returns
    -------

    vals: list
        list of numpy arrays

    offset_y:
        offset for the y-axis, if this is supposed to be in log10-scale

    y_label:
        Label for y axis
    """

    # get list of indices
    if start_indices is None:
        start_indices = np.array(range(len(vals)))
    else:
        # check whether list or maximum value
        start_indices = np.array(start_indices)

        # check, whether index set is not too big
        existing_indices = np.array(range(len(vals)))
        start_indices = np.intersect1d(start_indices, existing_indices)

    # reduce values to listed values
    vals = [val for i, val in enumerate(vals) if i in start_indices]

    # get the minimal value which should be plotted
    min_val = np.inf
    for val in vals:
        tmp_min = np.min(val[1, :])
        min_val = np.min([min_val, tmp_min])

    # check, whether offset can be used with this data
    if y_label == 'fval':
        offset_y = process_offset_y(offset_y, scale_y, min_val)
    else:
        offset_y = 0.

    if offset_y != 0:
        for val in vals:
            val[1, :] += offset_y * np.ones(val[1].shape)
            y_label = 'offsetted ' + y_label

    return vals, offset_y, y_label


def handle_options(ax: plt.Axes,
                   vals: List[np.ndarray],
                   ref: List[ReferencePoint],
                   y_limits: Union[float, np.ndarray],
                   offset_y: float):
    """
    Get the limits for the y-axis, plots the reference points, will do
    more at a later time point. This function is there to apply whatever
    kind of post-plotting transformations to the axis object.

    Parameters
    ----------

    ref:
        List of reference points for optimization results, containing et
        least a function value fval

    vals:
        list of numpy arrays of size 2 x number of values

    ax:
        Axes object to use.

    y_limits:
        maximum value to be plotted on the y-axis, or y-limits

    offset_y:
        offset for the y-axis, if this is supposed to be in log10-scale

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # handle y-limits
    ax = process_y_limits(ax, y_limits)

    # handle reference points
    if len(ref) > 0:
        # plot reference points
        # get length of longest trajectory
        max_len = 0
        for val in vals:
            max_len = np.max([max_len, val[0, -1]])

        for i_ref in ref:
            ax.plot([0, max_len],
                    [i_ref.fval + offset_y, i_ref.fval + offset_y],
                    '--', color=i_ref.color, label=i_ref.legend)

            # create legend for reference points
            if i_ref.legend is not None:
                ax.legend()

    return ax
