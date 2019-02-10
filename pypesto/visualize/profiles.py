import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .visualization import VisualizationOptions
from .visualization import handle_options


def profiles(result, fig=None, profile_indices=None, size=(18.5, 6.5),
             options=None, reference=None):
    """
    Plot classical 1D profile plot (using the posterior, e.g. Gaussian like
    profile)

    Parameters
    ----------

    result: pypesto.Result
        Optimization result obtained by 'optimize.py'

    fig: matplotlib.Figure, optional
        Figure object to use.

    profile_indices: list of integer values
        list of integer values specifying which profiles should be plotted

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    options: VisualizationOptions, optional
        Options specifying axes, colors and reference points

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    if profile_indices is None:
        profile_indices = \
            [i for i in range(0, len(result.profile_result.list[0]))]

    # extract ratio values values from result
    fvals = []
    for i_par in range(0, len(result.profile_result.list[0])):
        if i_par in profile_indices:
            tmp = np.array(
                [result.profile_result.list[0][i_par].x_path[i_par, :],
                 result.profile_result.list[0][i_par].ratio_path[:]])
        else:
            tmp = None
        fvals.append(tmp)

    # call lowlevel routine
    ax = profiles_lowlevel(fvals, fig, size)

    # parse and apply plotting options
    ref = handle_options(ax, options, reference)

    # plot reference points
    if ref is not None:
        ax = handle_refrence_points(ref, ax, fvals)

    return ax


def profiles_lowlevel(fvals, ax=None, size=(18.5, 6.5)):
    """
    Lowlevel routine for profile plotting, working with a list of arrays
    only, opening different axes objects in case

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    ax: list of matplotlib.Axes, optional
        list of axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # axes
    if ax is None:
        ax = []
        fig = plt.figure()
        fig.set_size_inches(*size)
    else:
        plt.axes(ax)
        fig = plt.gcf()

    if isinstance(fvals, list):
        n_fvals = 0
        for fval in enumerate(fvals):
            if fval is not None:
                n_fvals += 1
    else:
        n_fvals = 1
        fvals = [fvals]

    columns = np.ceil(np.sqrt(n_fvals))
    if n_fvals > columns * (columns - 1):
        rows = columns
    else:
        rows = columns - 1

    counter = 0
    for i_plot, fval in enumerate(fvals):

        if fval is not None:
            ax.append(fig.add_subplot(rows, columns, counter + 1))
            ax[counter] = profile_lowlevel(fval, ax[counter])

            # labels
            ax[counter].set_xlabel(f'Parameter {i_plot} value')
            if counter % columns == 0:
                ax[counter].set_ylabel('Log-posterior ratio')
            else:
                ax[counter].set_yticklabels([''])
            counter += 1

    return ax


def profile_lowlevel(fvals, ax=None, size=(18.5, 6.5)):
    """
    Lowlevel routine for plotting one profile, working with a numpy array only

    Parameters
    ----------

    fvals: numeric list or array
        Including values need to be plotted.

    ax: matplotlib.Axes, optional
        Axes object to use.

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    fvals = np.array(fvals)

    # axes
    if ax is None:
        ax = plt.subplots()[1]
        ax.set_xlabel('Parameter value')
        ax.set_ylabel('Log-posterior ratio')
        fig = plt.gcf()
        fig.set_size_inches(*size)

    # plot
    if fvals.size != 0:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(fvals[0, :], fvals[1, :], color=[.9, .2, .2, 1.])

    return ax


def handle_refrence_points(ref, ax, fvals):
    """
    Handle reference points.

    Parameters
    ----------

    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    ax: matplotlib.Axes, optional
        Axes object to use.
    """

    # get the parameters which have profiles plotted
    par_indices = []
    for i_plot, fval in enumerate(fvals):
        if fval is not None:
            par_indices.append(i_plot)

    # create set of colors for reference points
    ref_len = len(ref)
    colors = []
    ref_x = []
    for i_num, i_ref in enumerate(ref):
        colors.append([0., 0.5 * (1. + i_num / ref_len), 0., 0.9])
        ref_x.append(i_ref["x"])

    # loop over axes objects
    for i_par, i_ax in enumerate(ax):
        for i_ref, i_col in enumerate(colors):
            current_x = ref_x[i_ref][par_indices[i_par]]
            i_ax.plot([current_x, current_x], [0., 1.], color=i_col)

    return ax
