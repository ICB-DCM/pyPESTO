import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from typing import Sequence, Tuple, Union
from warnings import warn

from ..result import Result
from .reference_points import create_references, ReferencePoint
from .clust_color import assign_colors
from .misc import process_result_list


def profiles(results: Union[Result, Sequence[Result]],
             ax=None,
             profile_indices: Sequence[int] = None,
             size: Sequence[float] = (18.5, 6.5),
             reference: Union[ReferencePoint, Sequence[ReferencePoint]] = None,
             colors=None,
             legends: Sequence[str] = None,
             x_labels: Sequence[str] = None,
             profile_list_ids: Union[int, Sequence[int]] = 0,
             ratio_min: float = 0.,
             show_bounds: bool = False):
    """
    Plot classical 1D profile plot (using the posterior, e.g. Gaussian like
    profile)

    Parameters
    ----------
    results: list or pypesto.Result
        List of or single `pypesto.Result` after profiling.
    ax: list of matplotlib.Axes, optional
        List of axes objects to use.
    profile_indices: list of integer values
        List of integer values specifying which profiles should be plotted.
    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    reference: list, optional
        List of reference points for optimization results, containing at
        least a function value fval.
    colors: list, or RGBA, optional
        List of colors, or single color.
    legends: list or str, optional
        Labels for line plots, one label per result object.
    x_labels: list of str
        Labels for parameter value axes (e.g. parameter names).
    profile_list_ids: int or list of ints, optional
        Index or list of indices of the profile lists to be used for profiling.
    ratio_min:
        Minimum ratio below which to cut off.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    results, profile_list_ids, colors, legends = process_result_list_profiles(
        results, profile_list_ids, colors, legends)

    # get the parameter ids to be plotted
    profile_indices = process_profile_indices(results, profile_indices,
                                              profile_list_ids)

    # loop over results
    for i_result, result in enumerate(results):
        for i_profile_list, profile_list_id in enumerate(profile_list_ids):
            fvals = handle_inputs(result, profile_indices=profile_indices,
                                  profile_list=profile_list_id,
                                  ratio_min=ratio_min)

            # add x_labels for parameters
            if x_labels is None:
                x_labels = [name for name, fval in
                            zip(result.problem.x_names, fvals)
                            if fval is not None]

            # plot multiple results or profile runs into one figure?
            if len(results) == 1 and len(profile_list_ids) > 1:
                # multiple profile runs per axes object
                color_ind = i_profile_list
            else:
                # multiple results per axes object
                color_ind = i_result

            # call lowlevel routine
            ax = profiles_lowlevel(
                fvals=fvals, ax=ax, size=size, color=colors[color_ind],
                legend_text=legends[color_ind], x_labels=x_labels,
                show_bounds=show_bounds, lb_full=result.problem.lb_full,
                ub_full=result.problem.ub_full)

    # parse and apply plotting options
    ref = create_references(references=reference)

    # plot reference points
    ax = handle_reference_points(ref, ax, profile_indices)

    plt.tight_layout()

    return ax


def profiles_lowlevel(
        fvals, ax=None, size: Tuple[float, float] = (18.5, 6.5),
        color=None, legend_text: str = None, x_labels=None,
        show_bounds: bool = False, lb_full=None, ub_full=None):
    """
    Lowlevel routine for profile plotting, working with a list of arrays
    only, opening different axes objects in case

    Parameters
    ----------
    fvals: numeric list or array
        Values to plot.
    ax: list of matplotlib.Axes, optional
        List of axes object to use.
    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    color: RGBA, optional
        Color for profiles in plot.
    legend_text: str
        Label for line plots.
    legend_text: List[str]
        Label for line plots.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    lb_full:
        Lower bound.
    ub_full:
        Upper bound.

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
        n_fvals = len(fvals)
    else:
        n_fvals = 1
        fvals = [fvals]

    # number of non-trivial profiles
    n_profiles = sum((fval is not None for fval in fvals))

    # if axes already exists, we have to match profiles to axes
    if not create_new_ax:
        if n_fvals != len(ax) and n_profiles != len(ax):
            raise ValueError(
                "Number of axes does not match number of profiles. Stopping.")
        elif n_fvals == len(ax) and n_profiles != len(ax):
            # we may have some empty profiles, which we have to skip
            n_plots = n_fvals
        else:
            # n_profiles == len(ax):, we have exactly as many profiles as axes
            n_plots = n_profiles
    else:
        n_plots = n_profiles

    if lb_full is None:
        lb_full = [None] * len(fvals)
    if ub_full is None:
        ub_full = [None] * len(fvals)

    # compute number of columns and rows
    columns = np.ceil(np.sqrt(n_plots))
    if n_plots > columns * (columns - 1):
        rows = columns
    else:
        rows = columns - 1

    counter = 0
    for i_plot, (fval, lb, ub) in enumerate(zip(fvals, lb_full, ub_full)):
        # if we have empty profiles and more axes than profiles: skip
        if n_plots != n_fvals and fval is None:
            continue

        # handle legend
        if i_plot == 0:
            tmp_legend = legend_text
        else:
            tmp_legend = None

        # create or choose an axes object
        if create_new_ax:
            ax.append(fig.add_subplot(rows, columns, counter + 1))
        else:
            plt.axes(ax[counter])

        # plot if data
        if fval is not None:
            # run lowlevel routine for one profile
            ax[counter] = profile_lowlevel(
                fval, ax[counter], size=size, color=color,
                legend_text=tmp_legend, show_bounds=show_bounds, lb=lb, ub=ub)

        # labels
        if x_labels is None:
            ax[counter].set_xlabel(f'Parameter {i_plot}')
        else:
            ax[counter].set_xlabel(x_labels[counter])

        if counter % columns == 0:
            ax[counter].set_ylabel('Log-posterior ratio')
        else:
            ax[counter].set_yticklabels([''])

        # increase counter and cleanup legend
        counter += 1

    return ax


def profile_lowlevel(
        fvals, ax=None, size: Tuple[float, float] = (18.5, 6.5),
        color=None, legend_text: str = None, show_bounds: bool = False,
        lb: float = None, ub: float = None):
    """
    Lowlevel routine for plotting one profile, working with a numpy array only

    Parameters
    ----------
    fvals: numeric list or array
        Values to plot.
    ax: matplotlib.Axes, optional
        Axes object to use.
    size: tuple, optional
        Figure size (width, height) in inches. Is only applied when no ax
        object is specified.
    color: RGBA, optional
        Color for profiles in plot.
    legend_text: str
        Label for line plots.
    show_bounds:
        Whether to show, and extend the plot to, the lower and upper bounds.
    lb:
        Lower bound.
    ub:
        Upper bound.

    Returns
    -------
    ax: matplotlib.Axes
        The plot axes.
    """

    # parse input
    fvals = np.asarray(fvals)

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

    if show_bounds:
        ax.set_xlim([lb, ub])

    return ax


def handle_reference_points(ref, ax, profile_indices):
    """
    Handle reference points.

    Parameters
    ----------
    ref: list, optional
        List of reference points for optimization results, containing et
        least a function value fval
    ax: matplotlib.Axes, optional
        Axes object to use.
    profile_indices: list of integer values
        List of integer values specifying which profiles should be plotted.
    """

    if len(ref) > 0:
        # loop over axes objects
        for i_par, i_ax in enumerate(ax):
            for i_ref in ref:
                current_x = i_ref['x'][profile_indices[i_par]]
                i_ax.plot([current_x, current_x], [0., 1.],
                          color=i_ref.color, label=i_ref.legend)

            # create legend for reference points
            if i_ref.legend is not None:
                i_ax.legend()

    return ax


def handle_inputs(
        result: Result,
        profile_indices: Sequence[int],
        profile_list: int,
        ratio_min: float):
    """
    Retrieves the values of the profiles to be plotted later from a
    pypesto.ProfileResult object

    Parameters
    ----------
    result: pypesto.Result
        Profile result obtained by 'profile.py'.
    profile_indices: list of integer values
        List of integer values specifying which profiles should be plotted.
    profile_list: int, optional
        Index of the profile list to be used for profiling.

    Returns
    -------
    fvals: numeric list
        Including values that need to be plotted.
    """

    # extract ratio values values from result
    fvals = []
    for i_par in range(0, len(result.profile_result.list[profile_list])):
        if i_par in profile_indices and \
                result.profile_result.list[profile_list][i_par] is not None:
            xs = result.profile_result.list[profile_list][i_par]\
                .x_path[i_par, :]
            ratios = result.profile_result.list[profile_list][i_par]\
                .ratio_path[:]

            # constrain
            indices = np.where(ratios > ratio_min)
            xs = xs[indices]
            ratios = ratios[indices]

            fvals_for_par = np.array([xs, ratios])
        else:
            fvals_for_par = None
        fvals.append(fvals_for_par)

    return fvals


def process_result_list_profiles(results: Result,
                                 profile_list_ids: Sequence[int],
                                 colors: Sequence[np.array],
                                 legends: Union[str, list]) -> Sequence[int]:
    """
    assigns colors and legends to a list of results while taking care of the
    special cases for profile plotting

    Parameters
    ----------
    results: list or pypesto.Result
        List of or single `pypesto.Result` after profiling.
    profile_list_ids: int or list of ints, optional
        Index or list of indices of the profile lists to be used for profiling.
    colors: list of RGBA colors
        colors for
    legends: list of str
        Legends for plotting

    Returns
    -------
    profile_indices: list of integer values
        corrected list of integer values specifying which profiles should be
        plotted.
    """

    # ensure list of ids
    if isinstance(profile_list_ids, int):
        profile_list_ids = [profile_list_ids]

    # check if we have a single result
    if isinstance(results, list):
        if len(results) != 1:
            # if we have no single result, then use the standard api
            results, colors, legends = process_result_list(
                results, colors, legends)
            return results, profile_list_ids, colors, legends
    else:
        # a single results was provided, so make a list out of it
        results = [results]

    # If we have a single result, we may still have multiple profile_list_ids
    # which should be plotted separately: use profile_list_ids as results dummy
    _, colors, legends = process_result_list(
            profile_list_ids, colors, legends)

    return results, profile_list_ids, colors, legends


def process_profile_indices(
        results: Sequence[Result],
        profile_indices: Sequence[int],
        profile_list_ids: Union[int, Sequence[int]]):
    """
    Retrieves the indices of the parameter for which profiles should be
    plotted later from a list of pypesto.ProfileResult objects
    """

    # get all parameter indices, for which profiles were computed
    plottable_indices = set()
    for result in results:
        for profile_list_id in profile_list_ids:
            # get parameter indices, for which profiles were computed
            if profile_list_id < len(result.profile_result.list):
                tmp_indices = [
                    par_id for par_id, prof in
                    enumerate(result.profile_result.list[profile_list_id])
                    if prof is not None]
                # profile_indices should contain all parameter indices,
                # for which in at least one of the results a profile exists
                plottable_indices.update(tmp_indices)
    plottable_indices = sorted(plottable_indices)

    # get the profiles, which should be plotted and sanitize, if not plottable
    if profile_indices is None:
        profile_indices = list(plottable_indices)
    else:
        for ind in profile_indices:
            if ind not in plottable_indices:
                profile_indices.remove(ind)
                warn('Requested to plot profile for parameter index %i, '
                     'but profile has not been computed.' % ind)

    return profile_indices
