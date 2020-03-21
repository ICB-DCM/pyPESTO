import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from .reference_points import create_references
from .clust_color import assign_colors
from .misc import process_result_list


def profiles(results, ax=None, profile_indices=None, size=(18.5, 6.5),
             reference=None, colors=None, legends=None, profile_list_id=0):
    """
    Plot classical 1D profile plot (using the posterior, e.g. Gaussian like
    profile)

    Parameters
    ----------

    results: list or pypesto.Result
        list of pypesto.Result or single pypesto.Result

    ax: list of matplotlib.Axes, optional
        List of axes objects to use.

    profile_indices: list of integer values
        list of integer values specifying which profiles should be plotted

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    reference: list, optional
        List of reference points for optimization results, containing et
        least a function value fval

    colors: list, or RGBA, optional
        list of colors, or single color
        color or list of colors for plotting. If not set, clustering is done
        and colors are assigned automatically

    legends: list or str, optional
        Labels for line plots, one label per result object

    profile_list_id: int, optional
        index of the profile list to be used for profiling

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    (results, colors, legends) = process_result_list(results, colors, legends)

    # get the correct number of parameter indices, even if not the same in
    # all result objects
    if profile_indices is None:
        profile_indices = []
        for result in results:
            tmp_indices = [ind for ind in range(len(
                result.profile_result.list[profile_list_id]))]
            profile_indices = list(set().union(profile_indices, tmp_indices))

    # loop over results
    for j, result in enumerate(results):
        fvals = handle_inputs(result, profile_indices=profile_indices,
                              profile_list_id=profile_list_id)

        # call lowlevel routine
        ax = profiles_lowlevel(fvals=fvals, ax=ax, size=size,
                               color=colors[j], legend_text=legends[j])

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    ax = handle_reference_points(ref, ax, fvals)

    return ax


def profiles_lowlevel(fvals, ax=None, size=(18.5, 6.5), color=None,
                      legend_text=None):
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

    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified

    color: RGBA, optional
        color for profiles in plot.

    legend_text: str
        Label for line plots

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
        create_new_ax = True
    else:
        plt.axes(ax[0])
        fig = plt.gcf()
        create_new_ax = False

    # count number of necessary axes
    if isinstance(fvals, list):
        n_fvals = np.sum([1 for fval in fvals if fval is not None])
    else:
        n_fvals = 1
        fvals = [fvals]

    # if axes already exist: does the number of axes fit?
    if len(fvals) != len(ax) and not create_new_ax:
        raise ('Number of axes does not match number of profiles. Stopping.')

    # compute number of columns and rows
    columns = np.ceil(np.sqrt(n_fvals))
    if n_fvals > columns * (columns - 1):
        rows = columns
    else:
        rows = columns - 1

    counter = 0
    for i_plot, fval in enumerate(fvals):
        # handle legend
        if i_plot == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # plot if data
        if fval is not None:
            # create or choose an axes object
            if create_new_ax:
                ax.append(fig.add_subplot(rows, columns, counter + 1))
            else:
                plt.axes(ax[counter])

            # run lowlevel routine for one profile
            ax[counter] = profile_lowlevel(fval, ax[counter],
                                           size=size, color=color,
                                           legend_text=tmp_legend)

            # labels
            ax[counter].set_xlabel(f'Parameter {i_plot} value')
            if counter % columns == 0:
                ax[counter].set_ylabel('Log-posterior ratio')
            else:
                ax[counter].set_yticklabels([''])

            # increase counter and cleanup legend
            counter += 1
            tmp_legend = None

    return ax


def profile_lowlevel(fvals, ax=None, size=(18.5, 6.5), color=None,
                     legend_text=None):
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

    color: RGBA, optional
        color for profiles in plot.

    legend_text: str
        Label for line plots

    Returns
    -------

    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    fvals = np.array(fvals)

    # get colors
    color = assign_colors([1.], color)

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
        ax.plot(fvals[0, :], fvals[1, :], color=color[0], label=legend_text)

    if legend_text is not None:
        ax.legend()

    return ax


def handle_reference_points(ref, ax, fvals):
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

    if len(ref) > 0:
        # get the parameters which have profiles plotted
        par_indices = []
        for i_plot, fval in enumerate(fvals):
            if fval is not None:
                par_indices.append(i_plot)

        # loop over axes objects
        for i_par, i_ax in enumerate(ax):
            for i_ref in ref:
                current_x = i_ref['x'][par_indices[i_par]]
                i_ax.plot([current_x, current_x], [0., 1.],
                          color=i_ref.color, label=i_ref.legend)

            # create legend for reference points
            if i_ref.legend is not None:
                i_ax.legend()

    return ax


def handle_inputs(result, profile_indices, profile_list_id=0):
    """
    Retrieves the values of the profiles to be plotted later from a
    pypesto.ProfileResult object

    Parameters
    ----------

    result: pypesto.Result
        Profile result obtained by 'profile.py'

    profile_indices: list of integer values
        list of integer values specifying which profiles should be plotted

    profile_list_id: int, optional
        index of the profile list to be used for profiling

    Returns
    -------

    fvals: numeric list
        Including values need to be plotted.
    """

    # extract ratio values values from result
    fvals = []
    for i_par in range(0, len(result.profile_result.list[0])):
        if i_par in profile_indices:
            tmp = np.array([result.profile_result.list[
                profile_list_id][i_par].x_path[i_par, :],
                result.profile_result.list[profile_list_id][
                i_par].ratio_path[:]])
        else:
            tmp = None
        fvals.append(tmp)

    return fvals
