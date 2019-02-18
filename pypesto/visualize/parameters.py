import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .reference_points import create_references
from .clust_color import assign_colors
from .misc import handle_result_list


def parameters(results, ax=None, free_indices_only=True, lb=None, ub=None,
               size=None, reference=None, colors=None, legends=None):
    """
    Plot parameter values.

    Parameters
    ----------

    results: pypesto.Result or list
        Optimization result obtained by 'optimize.py' or list of those

    ax: matplotlib.Axes, optional
        Axes object to use.

    free_indices_only: bool, optional
        If True, only free parameters are shown. If
        False, also the fixed parameters are shown.

    lb, ub: ndarray, optional
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGB, optional
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

    # parse input
    (results, colors, legends) = handle_result_list(results, colors, legends)

    for j, result in enumerate(results):
        # handle results and bounds
        (lb, ub, x_labels, fvals, xs) = \
            handle_inputs(result=result, lb=lb, ub=ub,
                          free_indices_only=free_indices_only)

        # call lowlevel routine
        ax = parameters_lowlevel(xs=xs, fvals=fvals, lb=lb, ub=ub,
                                 x_labels=x_labels, ax=ax, size=size,
                                 colors=colors[j], legend_text=legends[j])

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    for i_ref in ref:
        ax = parameters_lowlevel([i_ref['x']], [i_ref['fval']], ax=ax,
                                 colors=i_ref['color'],
                                 legend_text=i_ref.legend)

    return ax


def parameters_lowlevel(xs, fvals, lb=None, ub=None, x_labels=None, ax=None,
                        size=None, colors=None, legend_text=None):
    """
    Plot parameters plot using list of parameters.

    Parameters
    ----------

    xs: nested list or array
        Including optimized parameters for each startpoint.
        Shape: (n_starts, dim).

    fvals: numeric list or array
        Function values. Needed to assign cluster colors.

    lb, ub: array_like, optional
        The lower and upper bounds.

    x_labels: array_like of str, optional
        Labels to be used for the parameters.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        see parameters

    colors: list of RGB
        One for each element in 'fvals'.

    legend_text: str
        Label for line plots

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    xs = np.array(xs)
    fvals = np.array(fvals)

    if size is None:
        # 0.5 inch height per parameter
        size = (18.5, xs.shape[1] / 2)

    if ax is None:
        ax = plt.subplots()[1]
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # assign colors
    colors = assign_colors(vals=fvals, colors=colors)

    # parameter indices
    parameters_ind = list(range(1, xs.shape[1] + 1))[::-1]

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j_x, x in reversed(list(enumerate(xs))):
        if j_x == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None
        ax.plot(x, parameters_ind,
                color=colors[j_x],
                marker='o',
                label=tmp_legend)

    plt.yticks(parameters_ind, x_labels)

    # draw bounds
    parameters_ind = np.array(parameters_ind).flatten()
    if lb is not None:
        ax.plot(lb.flatten(), parameters_ind, 'k--', marker='+')
    if ub is not None:
        ax.plot(ub.flatten(), parameters_ind, 'k--', marker='+')

    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter index')
    ax.set_title('Estimated parameters')

    return ax


def handle_inputs(result, free_indices_only, lb=None, ub=None):
    """
    Handle bounds and results.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.

    free_indices_only: bool, optional
        If True, only free parameters are shown. If
        False, also the fixed parameters are shown.

    lb, ub: ndarray, optional
        If not None, override result.problem.lb, problem.problem.ub.
        Dimension either result.problem.dim or result.problem.dim_full.

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # retrieve results
    fvals = result.optimize_result.get_for_key('fval')
    xs = result.optimize_result.get_for_key('x')

    # get bounds
    if lb is None:
        lb = result.problem.lb
    if ub is None:
        ub = result.problem.ub

    # get labels
    x_labels = result.problem.x_names

    # handle fixed and free indices
    if free_indices_only:
        for ix, x in enumerate(xs):
            xs[ix] = result.problem.get_reduced_vector(x)
        lb = result.problem.get_reduced_vector(lb)
        ub = result.problem.get_reduced_vector(ub)
        x_labels = [x_labels[int(i)] for i in result.problem.x_free_indices]
    else:
        lb = result.problem.get_full_vector(lb)
        ub = result.problem.get_full_vector(ub)

    return lb, ub, x_labels, fvals, xs
