import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .clust_color import assign_clustered_colors


def parameters(result, ax=None):
    """
    Plot parameter values.

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'.

    ax: matplotlib.Axes, optional
        Axes object to use.

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    fvals = result.optimize_result.get_for_key('fval')
    xs = result.optimize_result.get_for_key('x')
    lb = result.problem.lb
    ub = result.problem.ub

    return parameters_lowlevel(xs=xs, fvals=fvals, lb=lb, ub=ub,
                               x_labels=None, ax=ax)


def parameters_lowlevel(xs, fvals, lb=None, ub=None, x_labels=None, ax=None):
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

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    if ax is None:
        ax = plt.subplots()[1]

    # parse input
    xs = np.array(xs)
    fvals = np.array(fvals)

    if x_labels is not None:
        raise NotImplementedError("x_labels not implemented.")

    # assign color
    colors = assign_clustered_colors(fvals)

    # parameter indices
    parameters_ind = range(1, xs.shape[1] + 1)

    # plot parameters
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for j_x, x in reversed(list(enumerate(xs))):
        ax.plot(x, parameters_ind, color=colors[j_x], marker='o')

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
