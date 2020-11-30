import logging
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Sequence, Tuple, Union

from .misc import (
    rgba2rgb,
    LEN_RGB,
    RGBA_BLACK,
    RGBA_MIN,
    RGBA_MAX,
)
from ..result import Result
from ..sample import McmcPtResult, calculate_ci, \
    evaluate_samples, calculate_prediction_profiles

logger = logging.getLogger(__name__)


def sampling_fval_trace(
        result: Result,
        i_chain: int = 0,
        full_trace: bool = False,
        stepsize: int = 1,
        title: str = None,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None):
    """Plot log-posterior (=function value) over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    full_trace:
        Plot the full trace including warm up. Default: False.
    stepsize:
        Only one in `stepsize` values is plotted.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """

    # get data which should be plotted
    _, params_fval, _, _, _ = get_data_to_plot(result=result,
                                               i_chain=i_chain,
                                               stepsize=stepsize,
                                               full_trace=full_trace)

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    sns.set(style="ticks")
    kwargs = {'edgecolor': "w",  # for edge color
              'linewidth': 0.3,
              's': 10}
    if full_trace:
        kwargs['hue'] = "converged"
        if len(params_fval[kwargs['hue']].unique()) == 1:
            kwargs['palette'] = ["#477ccd"]
        elif len(params_fval[kwargs['hue']].unique()) == 2:
            kwargs['palette'] = ["#868686", "#477ccd"]
        kwargs['legend'] = False

    sns.scatterplot(x="iteration", y="logPosterior", data=params_fval,
                    ax=ax, **kwargs)

    if result.sample_result.burn_in is None:
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    if full_trace and _burn_in > 0:
        ax.axvline(_burn_in,
                   linestyle='--', linewidth=1.5,
                   color='k')

    ax.set_xlabel('iteration index')
    ax.set_ylabel('log-posterior')

    if title:
        ax.set_title(title)

    sns.despine()

    return ax


def sampling_prediction_profiles(result: Result,
                                 alpha: Sequence[int] = None,
                                 stepsize: int = 1,
                                 plot_type: str = 'states',
                                 title: str = None,
                                 size: Tuple[float, float] = None,
                                 ax: matplotlib.axes.Axes = None):
    """Plot MCMC-based prediction confidence intervals for the
    model states or observables. One or various confidence levels
    can be depicted.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    alpha:
        List of lower tail probabilities, defaults to 95% interval.
    stepsize:
        Only one in `stepsize` values is simulated for the intervals
        generation. Recommended for long MCMC chains. Defaults to 1.
    plot_type:
        Visualization mode for prediction intervals {‘states’, ‘observables’}.
        Defaults to ‘states’.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    axes:
        The plot axes.
    """
    if alpha is None:
        alpha = [95]

    # Evaluate prediction uncertainties
    evaluation = evaluate_samples(result, stepsize)

    # automatically sort values in decreasing order
    alpha_sorted = sorted(alpha, reverse=True)

    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(alpha_sorted) + 1)
    colors = [plt.cm.Blues_r(x) for x in evenly_spaced_interval]

    if plot_type == 'states':
        values = evaluation[1]
        yname = 'X'

    elif plot_type == 'observables':
        values = evaluation[0]
        yname = 'Y'

    # Number of observables/states
    nr_variables = values.shape[-1]
    ynames = [f'dummy_{i}' for i in range(nr_variables)]

    # set axes and figure
    if ax is None:
        # compute, how many rows and columns we need for the subplots
        num_row = int(np.round(np.sqrt(nr_variables)))
        num_col = int(np.ceil(nr_variables / num_row))

        fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=size)
    else:
        fig = ax.get_figure()

    axes = dict(zip(ynames, ax.flat))

    for i, level in enumerate(alpha_sorted):

        # Get upper and lower bounds for the confidence level
        lb, ub = calculate_prediction_profiles(values, alpha=level / 100)

        # Get the median
        _median = np.percentile(values, 50, axis=1)

        n_timepoints = _median.shape[1]
        n_conditions = _median.shape[0]

        # Loop over observables/states
        for j, iplot in enumerate(ynames):
            # Create array for X axis
            t = np.arange(n_timepoints)
            # Loop over experimental conditions
            for k in range(n_conditions):
                # Plot confidence region
                axes[iplot].fill_between(t,
                                         ub[k, :, j],
                                         lb[k, :, j],
                                         facecolor=colors[i],
                                         alpha=0.9,
                                         label=str(level) + '% CI')
                # Plot median
                axes[iplot].plot(t, _median[k, :, j],
                                 'k-', label='MCMC median')
                axes[iplot].set_ylabel(yname + '_' + str(j + 1))
                # Update X axis array
                t += n_timepoints

    if title:
        fig.suptitle(title)

    # create legend
    fig.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[iplot].legend(by_label.values(), by_label.keys(),
                       bbox_to_anchor=(1.05, 1))

    return axes


def sampling_prediction_profiles_conditions(
        result: Result,
        ci_levels: Union[int, Sequence[int]] = 95,
        step_size: int = 1,
        plot_type: str = 'states',
        title: str = None,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None,
        y_names: Sequence[str] = None,
        n_procs: int = 1,
        axis_label_padding: int = 30,
):
    """Plot MCMC-based prediction confidence intervals for the
    model states or observables. One or various confidence levels
    can be depicted. Plots are grouped by condition.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    ci_levels:
        List of lower tail probabilities, e.g. 95 for a 95% interval.
    step_size:
        Only one in `step_size` values is simulated for the intervals
        generation. Recommended for long MCMC chains.
    plot_type:
        Visualization mode for prediction intervals.
        Options: `'states'`, `'observables'`.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.
    y_names:
        Names for the plotted dependent variables.
    n_procs:
        The number of processors to use, to parallelize the evaluation of
        samples.
    axis_label_padding:
        Pixels between axis labels and plots.

    Returns
    -------
    axes:
        The plot axes.
    """
    if isinstance(ci_levels, int):
        ci_levels = [ci_levels]
    ci_levels = sorted(ci_levels, reverse=True)
    ci_levels_opacity = sorted(
        # min 30%, max 100%, opacity
        np.linspace(0.3 * RGBA_MAX, RGBA_MAX, len(ci_levels)),
        reverse=True,
    )
    if ax is not None:
        fig = ax.get_figure()
        if not isinstance(ax, np.ndarray):
            ax = np.array([[ax]])
    cmap = plt.cm.viridis
    cmap_min = RGBA_MIN
    cmap_max = 0.85*(RGBA_MAX - RGBA_MIN) + RGBA_MIN  # exclude yellows

    # Evaluate prediction uncertainties
    evaluation = evaluate_samples(result, step_size, n_procs=n_procs)

    values_options = {
        'observables': 0,
        'states': 1,
    }
    values = evaluation[values_options[plot_type]]
    n_variables = values.shape[-1]

    if y_names is None:
        y_names = [f'{plot_type}_{i}' for i in range(n_variables)]
    else:
        assert len(y_names) == n_variables

    # define colormap
    variables_color = [
        list(cmap(v))[:LEN_RGB]
        for v in np.linspace(cmap_min, cmap_max, n_variables)
    ]

    for i, level in enumerate(ci_levels):
        # Get upper and lower bounds for the CI level
        lb, ub = calculate_prediction_profiles(values, alpha=level/100)
        median = np.percentile(values, 50, axis=1)

        t = np.arange(median.shape[1])
        n_conditions = median.shape[0]

        if ax is None:
            n_row = int(np.round(np.sqrt(n_conditions)))
            n_col = int(np.ceil(n_conditions / n_row))
            fig, ax = plt.subplots(n_row, n_col, figsize=size, squeeze=False)

        for k in range(n_conditions):
            for j, y_name in enumerate(y_names):
                ax.flat[k].fill_between(
                    t,
                    ub[k, :, j],
                    lb[k, :, j],
                    facecolor=rgba2rgb(
                        variables_color[j] + [ci_levels_opacity[i]]),
                    label=f'{y_name}: MCMC {level} % CI',
                    lw=0,
                )
                ax.flat[k].plot(
                    t,
                    median[k, :, j],
                    'k-',
                    label=f'{y_name}: MCMC median'
                )

    if title:
        fig.suptitle(title)

    # Fake plots for legend line styles
    fake_data = [[0], [0]]
    lines_variables = np.array([
        # Assumes that the color for a variable is always the same, with
        # different opacity for different confidence interval levels.
        [y_name, Line2D(*fake_data, color=variables_color[index], lw=4)]
        for index, y_name in enumerate(y_names)
    ])
    lines_levels = np.array([
        # Assumes that different CI levels are represented as
        # difference opacities of the same color.
        [
            f'{level}% CI',
            Line2D(
                *fake_data,
                color=rgba2rgb([
                    *RGBA_BLACK[:LEN_RGB],
                    ci_levels_opacity[index]
                ]),
                lw=4
            )
        ]
        for index, level in enumerate(ci_levels)
    ] + [['Median', Line2D(*fake_data, color=RGBA_BLACK)]]
    )

    artist_padding = (
        axis_label_padding / (plt.gcf().get_size_inches()*fig.dpi)[0]
    )

    # CI level, and variable name, legends.
    legend_options_top_right = {
        'bbox_to_anchor': (1 + artist_padding, 1),
        'loc': 'upper left',
    }
    legend_options_bottom_right = {
        'bbox_to_anchor': (1 + artist_padding, 0),
        'loc': 'lower left',
    }
    legend_variables = ax.flat[n_col-1].legend(
        lines_variables[:, 1],
        lines_variables[:, 0],
        **legend_options_top_right,
        title=plot_type.capitalize(),
    )
    # Legend for CI levels
    ax.flat[-1].legend(
        lines_levels[:, 1],
        lines_levels[:, 0],
        **legend_options_bottom_right,
        title='MCMC',
    )
    fig.add_artist(legend_variables)

    # X and Y labels
    xmin = min(_ax.get_position().xmin for _ax in ax.flat)
    ymin = min(_ax.get_position().ymin for _ax in ax.flat)
    plt.text(
        0.5,
        ymin - artist_padding,
        'Time',
        ha='center',
        va='center',
        transform=fig.transFigure
    )
    plt.text(
        xmin - artist_padding,
        0.5,
        f'{plot_type.capitalize()} Values',
        ha='center',
        va='center',
        transform=fig.transFigure,
        rotation='vertical'
    )

    plt.tight_layout()
    return ax


def sampling_parameters_cis(
        result: Result,
        alpha: Sequence[int] = None,
        step: float = 0.05,
        show_median: bool = True,
        title: str = None,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None):
    """Plot MCMC-based parameter confidence intervals for
    one or various confidence levels.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    alpha:
        List of lower tail probabilities, defaults to 95% interval.
    step:
        Height of boxes for projectile plot, defaults to 0.05.
    show_median:
        Plot the median of the MCMC chain. Default: True.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    if alpha is None:
        alpha = [95]

    # automatically sort values in decreasing order
    alpha_sorted = sorted(alpha, reverse=True)
    # define colormap
    evenly_spaced_interval = np.linspace(0, 1, len(alpha_sorted))
    colors = [plt.cm.tab20c_r(x) for x in evenly_spaced_interval]
    # number of sampled parameters
    n_pars = result.sample_result.trace_x.shape[-1]

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    # loop over parameters
    for npar in range(n_pars):
        # initialize height of boxes
        _step = step
        # loop over confidence levels
        for n, level in enumerate(alpha_sorted):
            # extract percentile-based confidence intervals
            lb, ub = calculate_ci(result=result, alpha=level / 100)

            # assemble boxes for projectile plot
            x1 = [lb[npar], ub[npar]]
            y1 = [npar + _step, npar + _step]
            y2 = [npar - _step, npar - _step]
            # Plot boxes
            ax.fill(np.append(x1, x1[::-1]),
                    np.append(y1, y2[::-1]), color=colors[n],
                    label=str(level) + '% CI')

            if show_median:
                if n == len(alpha_sorted) - 1:
                    burn_in = result.sample_result.burn_in
                    converged = result.sample_result.trace_x[0, burn_in:, npar]
                    _median = np.median(converged)
                    ax.plot([_median, _median], [npar - _step, npar + _step],
                            'k-', label='MCMC median')

            # increment height of boxes
            _step += step

    ax.set_yticks(range(n_pars))
    ax.set_yticklabels(result.problem.x_names)
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Parameter name')

    if title:
        ax.set_title(title)

    plt.gca().invert_yaxis()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1))

    return ax


def sampling_parameters_trace(
        result: Result,
        i_chain: int = 0,
        par_indices: Sequence[int] = None,
        full_trace: bool = False,
        stepsize: int = 1,
        use_problem_bounds: bool = True,
        suptitle: str = None,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None):
    """Plot parameter values over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    par_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.
    full_trace:
        Plot the full trace including warm up. Default: False.
    stepsize:
        Only one in `stepsize` values is plotted.
    use_problem_bounds:
        Defines if the y-limits shall be the lower and upper bounds of
        parameter estimation problem.
    suptitle:
        Figure suptitle.
    size:
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result, i_chain=i_chain, stepsize=stepsize,
        full_trace=full_trace, par_indices=par_indices)

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    # set axes and figure
    if ax is None:
        fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=size)
    else:
        fig = ax.get_figure()

    axes = dict(zip(param_names, ax.flat))

    sns.set(style="ticks")
    kwargs = {'edgecolor': "w",  # for edge color
              'linewidth': 0.3,
              's': 10}

    if full_trace:
        kwargs['hue'] = "converged"
        if len(params_fval[kwargs['hue']].unique()) == 1:
            kwargs['palette'] = ["#477ccd"]
        elif len(params_fval[kwargs['hue']].unique()) == 2:
            kwargs['palette'] = ["#868686", "#477ccd"]
        kwargs['legend'] = False

    if result.sample_result.burn_in is None:
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    for idx, plot_id in enumerate(param_names):
        ax = axes[plot_id]

        ax = sns.scatterplot(x="iteration", y=plot_id, data=params_fval,
                             ax=ax, **kwargs)

        if full_trace and _burn_in > 0:
            ax.axvline(_burn_in,
                       linestyle='--', linewidth=1.5,
                       color='k')

        ax.set_xlabel('iteration index')
        ax.set_ylabel(param_names[idx])
        if use_problem_bounds:
            ax.set_ylim([theta_lb[idx], theta_ub[idx]])

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    sns.despine()

    return ax


def sampling_scatter(
        result: Result,
        i_chain: int = 0,
        stepsize: int = 1,
        suptitle: str = None,
        diag_kind: str = "kde",
        size: Tuple[float, float] = None):
    """Parameter scatter plot.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    stepsize:
        Only one in `stepsize` values is plotted.
    suptitle:
        Figure super title.
    diag_kind:
        Visualization mode for marginal densities {‘auto’, ‘hist’, ‘kde’, None}
    size:
        Figure size in inches.

    Returns
    -------
    ax:
        The plot axes.
    """

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, _ = get_data_to_plot(
        result=result, i_chain=i_chain, stepsize=stepsize)

    sns.set(style="ticks")

    ax = sns.pairplot(
        params_fval.drop(['logPosterior', 'iteration'], axis=1),
        diag_kind=diag_kind)

    if size is not None:
        ax.fig.set_size_inches(size)

    if suptitle:
        ax.fig.suptitle(suptitle)

    return ax


def sampling_1d_marginals(
        result: Result,
        i_chain: int = 0,
        par_indices: Sequence[int] = None,
        stepsize: int = 1,
        plot_type: str = 'both',
        bw: str = 'scott',
        suptitle: str = None,
        size: Tuple[float, float] = None):
    """
    Plot marginals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    par_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.
    stepsize:
        Only one in `stepsize` values is plotted.
    plot_type: {'hist'|'kde'|'both'}
        Specify whether to plot a histogram ('hist'), a kernel density estimate
        ('kde'), or both ('both').
    bw: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method.
    suptitle:
        Figure super title.
    size:
        Figure size in inches.

    Return
    --------
    ax: matplotlib-axes
    """

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub, param_names = get_data_to_plot(
        result=result, i_chain=i_chain,
        stepsize=stepsize, par_indices=par_indices)

    # compute, how many rows and columns we need for the subplots
    num_row = int(np.round(np.sqrt(nr_params)))
    num_col = int(np.ceil(nr_params / num_row))

    fig, ax = plt.subplots(num_row, num_col, squeeze=False, figsize=size)

    par_ax = dict(zip(param_names, ax.flat))
    sns.set(style="ticks")

    # fig, ax = plt.subplots(nr_params, figsize=size)[1]
    for idx, par_id in enumerate(param_names):
        if plot_type == 'kde':
            sns.kdeplot(params_fval[par_id], bw=bw, ax=par_ax[par_id])
        elif plot_type == 'hist':
            sns.distplot(
                params_fval[par_id], kde=False, rug=True, ax=par_ax[par_id])
        elif plot_type == 'both':
            sns.distplot(params_fval[par_id], rug=True, ax=par_ax[par_id])

        par_ax[par_id].set_xlabel(param_names[idx])
        par_ax[par_id].set_ylabel('Density')

    sns.despine()

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()

    return ax


def get_data_to_plot(
        result: Result, i_chain: int, stepsize: int,
        full_trace: bool = False, par_indices: Sequence[int] = None):
    """Get the data which should be plotted as a pandas.DataFrame.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot.
    stepsize:
        Only one in `stepsize` values is plotted.
    full_trace:
        Keep the full length of the chain. Default: False.
    par_indices: list of integer values
        List of integer values specifying which parameters to plot.
        Default: All parameters are shown.

    Returns
    -------
    nr_params:
        Number of parameters to be plotted.
    params_fval:
        Log posterior values to be plotted.
    theta_lb:
        Parameter lower bounds to be plotted.
    theta_ub:
        Parameter upper bounds to be plotted.
    param_names:
        Parameter names to be plotted.
    """
    # get parameters and fval results as numpy arrays
    arr_param = np.array(result.sample_result.trace_x[i_chain])

    if result.sample_result.burn_in is None:
        logger.warning("Burn in index not found in the results, "
                       "the full chain will be shown.\n"
                       "You may want to use, e.g., "
                       "'pypesto.sample.geweke_test'.")
        _burn_in = 0
    else:
        _burn_in = result.sample_result.burn_in

    # Burn in index
    if full_trace is False:
        burn_in = _burn_in
    else:
        burn_in = 0

    sample_result: McmcPtResult = result.sample_result

    # thin out by stepsize, from the index burn_in until end of vector
    arr_param = arr_param[np.arange(burn_in, len(arr_param), stepsize)]

    # invert sign for log posterior values
    arr_fval = - np.array(sample_result.trace_neglogpost[i_chain])
    indices = np.arange(burn_in, len(arr_fval), stepsize)
    arr_fval = arr_fval[indices]
    theta_lb = result.problem.lb
    theta_ub = result.problem.ub

    # get parameter names from all non-fixed parameters
    param_names = result.problem.get_reduced_vector(result.problem.x_names)

    # transform ndarray to pandas for the use of seaborn
    pd_params = pd.DataFrame(arr_param, columns=param_names)
    pd_fval = pd.DataFrame(data=arr_fval, columns=['logPosterior'])

    pd_iter = pd.DataFrame(data=indices, columns=['iteration'])

    if full_trace:
        converged = np.zeros((len(arr_fval)))
        converged[_burn_in:] = 1
        pd_conv = pd.DataFrame(data=converged, columns=['converged'])

        params_fval = pd.concat(
            [pd_params, pd_fval, pd_iter, pd_conv], axis=1, ignore_index=False)
    else:
        params_fval = pd.concat(
            [pd_params, pd_fval, pd_iter], axis=1, ignore_index=False)

    # some global parameters
    nr_params = arr_param.shape[1]  # number of parameters

    if par_indices is not None:
        param_names = params_fval.columns.values[par_indices]
        nr_params = len(par_indices)
    else:
        param_names = params_fval.columns.values[0:nr_params]

    return nr_params, params_fval, theta_lb, theta_ub, param_names
