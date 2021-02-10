import logging
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Sequence, Tuple, Union

from .misc import (
    rgba2rgb,
    LEN_RGB,
    RGBA_BLACK,
    RGBA_MIN,
    RGBA_MAX,
)
from ..result import Result
from ..sample import McmcPtResult, calculate_ci

from ..ensemble import (
    get_percentile_label,
    EnsemblePrediction,
    MEDIAN,
)

logger = logging.getLogger(__name__)


# TODO observable here refers to the output of an ensemble prediction, which
#      means that observable can refer to either model state or observable
#      (user-specified).
OBSERVABLE = 'observable'
CONDITION = 'condition'


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


def _get_percentile_cutoffs(percentiles: float):
    half_difference = (100 - percentiles)/2
    return {'lower': half_difference, 'upper': 100 - half_difference}


def sampling_prediction_trajectories(
        ensemble_prediction: EnsemblePrediction,
        percentiles: Union[float, Sequence[float]],
        title: str = None,
        size: Tuple[float, float] = None,
        axes: matplotlib.axes.Axes = None,
        labels: Dict[str, str] = None,
        axis_label_padding: int = 30,
        groupby: str = CONDITION,
        condition_gap: float = 0.01,
):
    """Plot MCMC-based prediction confidence intervals for the
    model states or observables. One or various confidence levels
    can be depicted. Plots are grouped by condition.

    Note that references to observables here corresponds to the meaning of an
    observable in the pyPESTO predictions code, which means an observable could
    refer to many things, including model states or observables, dependent on
    how the prediction is processed. FIXME rename

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    percentiles:
        Credibility levels, e.g. `[95]` for a 95% credibility interval. These
        are split symmetrically, e.g. [95] corresponds to plotting values
        between the 2.5% and 97.5% percentiles, and are equivalent to highest
        density regions.
    title:
        Axes title.
    size: ndarray
        Figure size in inches.
    axes:
        Axes object to use.
    labels:
        Keys should be ensemble observable IDs, values should be the desired
        label for that observable. Defaults to observable IDs.
    axis_label_padding:
        Pixels between axis labels and plots.
    groupby:
        Group plots by `OBSERVABLE` or `CONDITION`. TODO single constants file?
    condition_gap:
        Gap between conditions when `groupby == CONDITION`.

    Returns
    -------
    axes:
        The plot axes.
    """
    if labels is None:
        labels = {}
    if len(list(percentiles)) == 1:
        percentiles = list(percentiles)
    percentiles = sorted(percentiles, reverse=True)
    percentiles_cutoffs_mapping = {
        p: _get_percentile_cutoffs(p)
        for p in percentiles
    }
    percentile_cutoffs = [
        cutoff
        for pair in percentiles_cutoffs_mapping.values()
        for cutoff in pair.values()
    ]

    ensemble_prediction.compute_summary(percentiles_list=percentile_cutoffs)

    summary = ensemble_prediction.prediction_summary

    all_condition_ids = [
        prediction.condition_ids
        for prediction in ensemble_prediction.prediction_summary.values()
    ]
    # All prediction results must predict for the same set of conditions.
    # Can support different conditions later.
    if not (
            np.array([
                set(condition_ids) == set(all_condition_ids[0])
                for condition_ids in all_condition_ids
            ]).all()
    ):
        raise KeyError('All predictions must have the same set of conditions.')
    condition_ids = all_condition_ids[0]

    observable_ids = sorted({
        observable_id
        for prediction in ensemble_prediction.prediction_summary.values()
        for condition in prediction.conditions
        for observable_id in condition.observable_ids
    })

    # Set default labels.
    labels = {
        k: (labels[k] if k in labels else k)
        for k in condition_ids + observable_ids
    }

    if groupby == CONDITION:
        n_variables = len(observable_ids)
        y_names = observable_ids
        n_subplots = len(condition_ids)
    elif groupby == OBSERVABLE:
        n_variables = len(condition_ids)
        y_names = condition_ids
        n_subplots = len(observable_ids)
    else:
        raise ValueError(f'Unsupported groupby value: {groupby}')

    percentiles_opacity = sorted(
        # min 30%, max 100%, opacity
        np.linspace(0.3 * RGBA_MAX, RGBA_MAX, len(percentiles)),
        reverse=True,
    )
    if axes is not None:
        fig = axes.get_figure()
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
    cmap = plt.cm.viridis
    cmap_min = RGBA_MIN
    cmap_max = 0.85*(RGBA_MAX - RGBA_MIN) + RGBA_MIN  # exclude yellows

    # define colormap
    variables_color = [
        list(cmap(v))[:LEN_RGB]
        for v in np.linspace(cmap_min, cmap_max, n_variables)
    ]

    if axes is None:
        n_row = int(np.round(np.sqrt(n_subplots)))
        n_col = int(np.ceil(n_subplots / n_row))
        fig, axes = plt.subplots(n_row, n_col, figsize=size, squeeze=False)
    elif len(axes.flat) < n_subplots:
        raise ValueError(
            'Provided `ax` contains insufficient subplots. At least '
            f'{n_subplots} are required.'
        )

    def get_statistic_data(
            statistic: str,
            condition_index: int,
            observable_id: str,
    ) -> Tuple[Sequence[float], Sequence[float]]:
        """Get statistic-, condition-, and observable-specific data.

        Parameters
        ----------
        statistic:
            The predicted statistic, e.g. `MEDIAN` or
            `get_percentile_label(95)`.
        condition_index:
            Select data for a specific condition by its index in a
            `PredictionResult.condition_ids` object.
        observable_id:
            Select data for a specific observable by its ID.

        Returns
        -------
        Predicted values and their corresponding time points. A tuple of two
        sequences, where the first sequence is time points, and the second
        sequence is predicted values at the corresponding time points.
        """
        condition_result = summary[statistic].conditions[condition_index]
        t = condition_result.timepoints
        observable_index = condition_result.observable_ids.index(observable_id)
        y = condition_result.output[:, observable_index]
        return (t, y)

    if groupby == CONDITION:
        # Each subplot has all data for a single condition.
        for condition_index, condition_id in enumerate(condition_ids):
            ax = axes.flat[condition_index]
            ax.set_title(f'Condition: {labels[condition_id]}')
            # Each subplot has all data for all condition-specific observables.
            for observable_index, observable_id in enumerate(observable_ids):
                ax.plot(
                    *get_statistic_data(
                        MEDIAN,
                        condition_index,
                        observable_id
                    ),
                    'k-',
                )
                for percentile_index, percentile in enumerate(percentiles):
                    lower_label = get_percentile_label(
                        percentiles_cutoffs_mapping[percentile]['lower']
                    )
                    upper_label = get_percentile_label(
                        percentiles_cutoffs_mapping[percentile]['upper']
                    )
                    t1, lower_data = get_statistic_data(
                        lower_label,
                        condition_index,
                        observable_id,
                    )
                    _, upper_data = get_statistic_data(
                        upper_label,
                        condition_index,
                        observable_id,
                    )
                    ax.fill_between(
                        t1,
                        lower_data,
                        upper_data,
                        facecolor=rgba2rgb(
                            variables_color[observable_index]
                            + [percentiles_opacity[percentile_index]]
                        ),
                        lw=0,
                    )
    elif groupby == OBSERVABLE:
        # Each subplot has all data for a single observable.
        for observable_index, observable_id in enumerate(observable_ids):
            t0 = 0
            ax = axes.flat[observable_index]
            ax.set_title(f'Trajectory: {labels[observable_id]}')
            # Each subplot is divided by conditions, with vertical lines.
            for condition_index, _condition_id in enumerate(condition_ids):
                if condition_index != 0:
                    ax.axvline(
                        t0,
                        linewidth=2,
                        color='k',
                    )

                t_max = t0
                t_median, y_median = get_statistic_data(
                    MEDIAN,
                    condition_index,
                    observable_id,
                )
                t_median_shifted = t_median + t0
                ax.plot(
                    t_median_shifted,
                    y_median,
                    'k-',
                )
                t_max = max(t_max, *t_median_shifted)
                for percentile_index, percentile in enumerate(percentiles):
                    lower_label = get_percentile_label(
                        percentiles_cutoffs_mapping[percentile]['lower']
                    )
                    upper_label = get_percentile_label(
                        percentiles_cutoffs_mapping[percentile]['upper']
                    )
                    t_lower, lower_data = get_statistic_data(
                        lower_label,
                        condition_index,
                        observable_id,
                    )
                    t_upper, upper_data = get_statistic_data(
                        upper_label,
                        condition_index,
                        observable_id,
                    )
                    t_lower_shifted = t_lower + t0
                    t_upper_shifted = t_upper + t0
                    # Timepoints must match, or `upper_data` will be plotted at
                    # some incorrect time points.
                    assert (np.array(t_lower) == np.array(t_upper)).all()
                    ax.fill_between(
                        t_lower_shifted,
                        lower_data,
                        upper_data,
                        facecolor=rgba2rgb(
                            variables_color[condition_index]
                            + [percentiles_opacity[percentile_index]]
                        ),
                        lw=0,
                    )
                    t_max = max(t_max, *t_lower_shifted, *t_upper_shifted)
                t0 = t_max

    if title:
        fig.suptitle(title)

    # Fake plots for legend line styles
    fake_data = [[0], [0]]
    lines_variables = np.array([
        # Assumes that the color for a variable is always the same, with
        # different opacity for different confidence interval levels.
        [
            labels[y_name],
            Line2D(*fake_data, color=variables_color[index], lw=4)
        ]
        for index, y_name in enumerate(y_names)
    ])
    lines_levels = np.array([
        # Assumes that different CI levels are represented as
        # different opacities of the same color.
        [
            f'{level}% CI',
            Line2D(
                *fake_data,
                color=rgba2rgb([
                    *RGBA_BLACK[:LEN_RGB],
                    percentiles_opacity[index]
                ]),
                lw=4
            )
        ]
        for index, level in enumerate(percentiles)
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
    legend_titles = {
        OBSERVABLE: 'Conditions',
        CONDITION: 'Trajectories',
    }
    legend_variables = axes.flat[n_col-1].legend(
        lines_variables[:, 1],
        lines_variables[:, 0],
        **legend_options_top_right,
        title=legend_titles[groupby],
    )
    # Legend for CI levels
    axes.flat[-1].legend(
        lines_levels[:, 1],
        lines_levels[:, 0],
        **legend_options_bottom_right,
        title='MCMC',
    )
    fig.add_artist(legend_variables)

    # X and Y labels
    xmin = min(ax.get_position().xmin for ax in axes.flat)
    ymin = min(ax.get_position().ymin for ax in axes.flat)
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
        'Simulated values',
        ha='center',
        va='center',
        transform=fig.transFigure,
        rotation='vertical'
    )

    # plt.tight_layout()  # Ruins layout for `groupby == OBSERVABLE`.
    return axes


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
