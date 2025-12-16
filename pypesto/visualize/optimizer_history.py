import logging
import warnings
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..C import (
    COLOR,
    TRACE_X_STEPS,
    TRACE_X_TIME,
    TRACE_Y_FVAL,
    TRACE_Y_GRADNORM,
)
from ..history import HistoryBase
from ..result import Result
from .clust_color import assign_colors
from .misc import process_offset_y, process_result_list, process_y_limits
from .reference_points import ReferencePoint, create_references

logger = logging.getLogger(__name__)


def optimizer_history(
    results: Result | list[Result],
    ax: plt.Axes | None = None,
    size: tuple = (18.5, 10.5),
    trace_x: str = TRACE_X_STEPS,
    trace_y: str = TRACE_Y_FVAL,
    scale_y: str = "log10",
    offset_y: float | None = None,
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    y_limits: float | list[float] | np.ndarray | None = None,
    start_indices: int | list[int] | None = None,
    reference: ReferencePoint
    | dict
    | list[ReferencePoint]
    | list[dict]
    | None = None,
    legends: str | list[str] | None = None,
) -> plt.Axes:
    """
    Plot history of optimizer.

    Can plot either the history of the cost function or of the gradient
    norm, over either the optimizer steps or the computation time.

    Parameters
    ----------
    results:
        Optimization result obtained by 'optimize.py' or list of those
    ax:
        Axes object to use.
    size:
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified
    trace_x:
        What should be plotted on the x-axis?
        Possibilities: TRACE_X
        Default: TRACE_X_STEPS
    trace_y:
        What should be plotted on the y-axis?
        Possibilities: TRACE_Y_FVAL, TRACE_Y_GRADNORM
        Default: TRACE_Y_FVAl
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    offset_y:
        Offset for the y-axis-values, as these are plotted on a log10-scale
        Will be computed automatically if necessary
    colors:
        Color recognized by matplotlib or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically
    y_limits:
        maximum value to be plotted on the y-axis, or y-limits
    start_indices:
        list of integers specifying the multistart to be plotted or
        int specifying up to which start index should be plotted
    reference:
        List of reference points for optimization results, containing at
        least a function value fval
    legends:
        Labels for line plots, one label per result object

    Returns
    -------
    ax:
        The plot axes.
    """
    if isinstance(start_indices, int):
        start_indices = list(range(start_indices))

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    for j, result in enumerate(results):
        # extract cost function values from result
        vals = get_trace(result, trace_x, trace_y)

        # compute the necessary offset for the y-axis
        (vals, offset_y) = get_vals(
            vals, scale_y, offset_y, trace_y, start_indices
        )

        x_label, y_label = get_labels(trace_x, trace_y, offset_y)

        # call lowlevel plot routine
        ax = optimizer_history_lowlevel(
            vals,
            scale_y=scale_y,
            ax=ax,
            colors=colors[j],
            size=size,
            x_label=x_label,
            y_label=y_label,
            legend_text=legends[j],
        )

    # parse and apply plotting options
    ref = create_references(references=reference)

    # handle options
    ax = handle_options(ax, vals, trace_y, ref, y_limits, offset_y)

    return ax


def optimizer_history_lowlevel(
    vals: list[np.ndarray],
    scale_y: str = "log10",
    colors: COLOR | list[COLOR] | np.ndarray | None = None,
    ax: plt.Axes | None = None,
    size: tuple = (18.5, 10.5),
    x_label: str = "Optimizer steps",
    y_label: str = "Objective value",
    legend_text: str | None = None,
) -> plt.Axes:
    """
    Plot optimizer history using list of numpy arrays.

    Parameters
    ----------
    vals:
        list of 2xn-arrays (x_values and y_values of the trace)
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    colors:
        Color recognized by matplotlib or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically
    ax:
        Axes object to use.
    size:
        see waterfall
    x_label:
        label for x-axis
    y_label:
        label for y-axis
    legend_text:
        Label for line plots

    Returns
    -------
    ax:
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
            # val is already an numpy array
            val = np.asarray(val)
            fvals.append(val[1, -1])
    else:
        # convert to a list of numpy arrays
        vals = np.asarray(vals)
        if vals.shape[0] != 2 or vals.ndim != 2:
            raise ValueError(
                "If numpy array is passed directly to lowlevel "
                "routine of optimizer_history, shape needs to "
                "be 2 x n."
            )
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
        if scale_y == "log10":
            ax.semilogy(val[0, :], val[1, :], color=color, label=tmp_legend)
        else:
            ax.plot(val[0, :], val[1, :], color=color, label=tmp_legend)

    # set labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Optimizer history")
    if legend_text is not None:
        ax.legend()

    return ax


def get_trace(
    result: Result, trace_x: str | None, trace_y: str | None
) -> list[np.ndarray]:
    """
    Get the values of the optimizer trace from the pypesto.Result object.

    Parameters
    ----------
    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.
    trace_x: str, optional
        What should be plotted on the x-axis?
        Possibilities: TRACE_X
        Default: TRACE_X_STEPS
    trace_y: str, optional
        What should be plotted on the y-axis?
        Possibilities: TRACE_Y_FVAL, TRACE_Y_GRADNORM
        Default: TRACE_Y_FVAL

    Returns
    -------
    vals:
        list of (x,y)-values.
    x_label:
        label for x-axis to be plotted later.
    y_label:
        label for y-axis to be plotted later.
    """
    # get data frames
    histories: list[HistoryBase] = result.optimize_result.history

    vals = []

    for history in histories:
        options = history.options
        if trace_y == TRACE_Y_GRADNORM:
            # retrieve gradient trace, if saved
            if not options.trace_record or not options.trace_record_grad:
                raise ValueError("No gradient trace has been recorded.")
            grads = history.get_grad_trace()
            indices = [
                i
                for i, val in enumerate(grads)
                if val is not None and np.isfinite(val).all()
            ]

            grads = np.array([grads[i] for i in indices])

            # Get gradient trace, compute norm
            y_vals = np.linalg.norm(grads, axis=1)

        else:  # trace_y == TRACE_Y_FVAL:
            if not options.trace_record:
                raise ValueError("No function value trace has been recorded.")
            fvals = history.get_fval_trace()
            indices = [
                i
                for i, val in enumerate(fvals)
                if val is not None and np.isfinite(val)
            ]

            y_vals = np.array([fvals[i] for i in indices])

        # retrieve values from dataframe
        if trace_x == TRACE_X_TIME:
            times = np.array(history.get_time_trace())
            x_vals = times[indices]

        else:  # trace_x == TRACE_X_STEPS:
            x_vals = np.array(list(range(len(indices))))

        # if the trace is empty, skip
        if len(x_vals) == 0:
            continue

        # write down values
        vals.append(np.vstack([x_vals, y_vals]))

    return vals


def get_vals(
    vals: list[np.ndarray],
    scale_y: str | None,
    offset_y: float,
    trace_y: str,
    start_indices: Iterable[int],
) -> tuple[list[np.ndarray], float]:
    """
    Postprocess the values of the optimization history.

    Depending on the options set by the user (e.g. scale_y, offset_y,
    start_indices).

    Parameters
    ----------
    vals:
        list of numpy arrays of dimension 2 x len(start_indices)
    scale_y:
        May be logarithmic or linear ('log10' or 'lin')
    offset_y:
        offset for the y-axis, as this is supposed to be in log10-scale
    trace_y:
        What should be plotted on the y-axis
    start_indices:
        list of integers specifying the multi start indices to be plotted

    Returns
    -------
    vals:
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
    if trace_y == TRACE_Y_FVAL:
        offset_y = process_offset_y(offset_y, scale_y, min_val)
    else:
        offset_y = 0.0

    if offset_y != 0:
        for val in vals:
            val[1, :] += offset_y * np.ones(val[1].shape)

    return vals, offset_y


def get_labels(trace_x: str, trace_y: str, offset_y: float) -> tuple[str, str]:
    """
    Generate labels for x and y axes of the history plot.

    Parameters
    ----------
    trace_x:
        What should be plotted on the x-axis. Possible values: TRACE_X.
    trace_y:
        What should be plotted on the y-axis.
        Possible values: TRACE_Y_FVAL, TRACE_Y_GRADNORM.
    offset_y:
        Offset for the y-axis-values.
    Returns
    -------
    labels for x and y axes
    """
    x_label = ""
    y_label = ""

    if trace_x == TRACE_X_TIME:
        x_label = "Computation time [s]"
    else:
        x_label = "Optimizer steps"

    if trace_y == TRACE_Y_GRADNORM:
        y_label = "Gradient norm"
    else:
        y_label = "Objective value"

    if offset_y != 0:
        y_label = "Offsetted " + y_label.lower()

    return x_label, y_label


def handle_options(
    ax: plt.Axes,
    vals: list[np.ndarray],
    trace_y: str,
    ref: list[ReferencePoint],
    y_limits: float | np.ndarray | None,
    offset_y: float,
) -> plt.Axes:
    """
    Apply post-plotting transformations to the axis object.

    Get the limits for the y-axis, plots the reference points, will do
    more at a later time point.

    Parameters
    ----------
    ref:
        List of reference points for optimization results, containing et
        least a function value fval
    vals:
        list of numpy arrays of size 2 x number of values
    trace_y:
        What should be plotted on the x-axis.
    ax:
        Axes object to use.
    y_limits:
        maximum value to be plotted on the y-axis, or y-limits
    offset_y:
        offset for the y-axis, if this is supposed to be in log10-scale

    Returns
    -------
    ax:
        The plot axes.
    """
    # handle y-limits
    ax = process_y_limits(ax, y_limits)

    if trace_y == TRACE_Y_FVAL:
        # handle reference points
        if len(ref) > 0:
            # plot reference points
            # get length of longest trajectory
            max_len = 0
            for val in vals:
                max_len = np.max([max_len, val[0, -1]])

            for i_ref in ref:
                ax.plot(
                    [0, max_len],
                    [i_ref.fval + offset_y, i_ref.fval + offset_y],
                    "--",
                    color=i_ref.color,
                    label=i_ref.legend,
                )

                # create legend for reference points
                if i_ref.legend is not None:
                    ax.legend()
    else:
        logger.warning(
            f"Reference point is currently only implemented for trace_y == "
            f"{TRACE_Y_FVAL} and will not be plotted for trace_y == {trace_y}."
        )

    return ax


def monotonic_history(
    histories: list[HistoryBase],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the overall monotonically decreasing history from multiple histories.

    Parameters
    ----------
    histories:
        List of histories to merge.
        The histories are expected to have time and function value traces,
        and that the function values are monotonically decreasing within each
        history.

    Returns
    -------
    t:
        Time points of overall history.
    fx:
        Objective values of overall history.
    """
    if len(histories) == 0:
        warnings.warn("No histories to process.", stacklevel=2)
        return np.array([], dtype="float"), np.array([], dtype="float")

    for history in histories:
        fvals = history.get_fval_trace()
        if not np.all(fvals[:-1] >= fvals[1:]):
            raise ValueError(
                "Each history is expected to have monotonically "
                "decreasing function values."
            )

    # merge results
    t = np.hstack([history.get_time_trace() for history in histories])
    fx = np.hstack([history.get_fval_trace() for history in histories])
    time_order = np.argsort(t)
    t = t[time_order]
    fx = fx[time_order]

    # get the monotonously decreasing sequence
    monotone = np.where(fx == np.fmin.accumulate(fx))[0]
    t_mono, fx_mono = t[monotone], fx[monotone]

    # remove duplicates
    # convert to np.recarray for multi-level sorting
    # so that we can use np.unique later to remove duplicates
    records = np.rec.fromarrays([t_mono, fx_mono], names="t,fx")
    records.sort(order=["t", "fx"])
    t_mono, unique_idx = np.unique(records.t, return_index=True, sorted=True)
    fx_mono = records.fx[unique_idx]

    # extend from last decrease to last timepoint
    if t_mono[-1] != (t_max := t.max()):
        t_mono = np.append(t_mono, [t_max])
        fx_mono = np.append(fx_mono, [fx_mono.min()])

    return t_mono, fx_mono


def sacess_history(
    histories: list[HistoryBase],
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot `SacessOptimizer` history.

    Plot the history of the best objective values for each
    :class:`pypesto.optimize.ess.sacess.SacessOptimizer`
    worker over computation time as step splot.

    Parameters
    ----------
    histories:
        List of histories from different workers as obtained from
        :attr:`pypesto.optimize.ess.sacess.SacessOptimizer.histories`.
    ax:
        Axes object to use.

    Returns
    -------
    The plot axes. `ax` or a new axes if `ax` was `None`.
    """
    ax = ax or plt.subplot()
    if len(histories) == 0:
        warnings.warn("No histories to plot.", stacklevel=2)

    # plot overall minimum
    t_overall, fx_overall = monotonic_history(histories)
    ax.step(
        t_overall,
        fx_overall,
        linestyle="dotted",
        color="grey",
        where="post",
        label="overall",
        alpha=0.8,
    )

    # plot steps of individual workers
    for worker_idx, history in enumerate(histories):
        x, y = history.get_time_trace(), history.get_fval_trace()
        if len(x) == 0:
            warnings.warn(f"No trace for worker #{worker_idx}.", stacklevel=2)
            continue
        # extend from last decrease to last timepoint
        x = np.append(x, [np.max(t_overall)])
        y = np.append(y, [np.min(y)])
        lines = ax.step(
            x, y, ".-", where="post", label=f"worker {worker_idx}", alpha=0.8
        )
        # Plot last point without marker, unless we actually had an improvement there.
        # The time point of the overall last improvement is appended to all histories,
        # even if redundant, so we can just skip the marker for the last point.
        for line in lines:
            line.set_markevery([True] * (len(x) - 1) + [False])

    ax.legend()
    ax.set_xlabel("time (s)")
    ax.set_ylabel("fval")
    ax.set_title("SacessOptimizer convergence")
    return ax
