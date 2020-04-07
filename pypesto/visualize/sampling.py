import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Tuple

from ..result import Result
from ..sample import McmcPtResult


def sampling_fval_trace(
        result: Result,
        i_chain: int = 0,
        burn_in: int = None,
        stepsize: int = 1,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None):
    """Plot log-posterior (=function value) over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    burn_in:
        Index after burn-in phase, thus also the burn-in length.
    stepsize:
        Only one in `stepsize` values is plotted.
    size: ndarray
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    # TODO: get burn_in from results object
    if burn_in is None:
        burn_in = 0

    # get data which should be plotted
    _, params_fval, _, _ = get_data_to_plot(
        result=result, i_chain=i_chain, burn_in=burn_in, stepsize=stepsize)

    # set axes and figure
    if ax is None:
        _, ax = plt.subplots(figsize=size)

    sns.set(style="ticks")
    kwargs = {'edgecolor': "w",  # for edge color
              'linewidth': 0.3,
              's': 10}
    sns.scatterplot(x="iteration", y="logPosterior", data=params_fval,
                    ax=ax, **kwargs)

    ax.set_xlabel('iteration index')
    ax.set_ylabel('log-posterior')
    if i_chain > 1:
        ax.set_title(f'Temperature chain: {i_chain}')

    sns.despine()

    return ax


def sampling_parameters_trace(
        result: Result,
        i_chain: int = 0,
        burn_in: int = None,
        stepsize: int = 1,
        use_problem_bounds: bool = True,
        size: Tuple[float, float] = None,
        ax: matplotlib.axes.Axes = None):
    """Plot parameter values over iterations.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    burn_in:
        Index after burn-in phase, thus also the burn-in length.
    stepsize:
        Only one in `stepsize` values is plotted.
    use_problem_bounds:
        Defines if the y-limits shall be the lower and upper bounds of
        parameter estimation problem.
    size:
        Figure size in inches.
    ax:
        Axes object to use.

    Returns
    -------
    ax:
        The plot axes.
    """
    # TODO: get burn_in from results object
    if burn_in is None:
        burn_in = 0

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = get_data_to_plot(
        result=result, i_chain=i_chain, burn_in=burn_in, stepsize=stepsize)

    param_names = params_fval.columns.values[0:nr_params]

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

    for idx, plot_id in enumerate(param_names):
        ax = axes[plot_id]
        ax = sns.scatterplot(x="iteration", y=plot_id, data=params_fval, ax=ax,
                             **kwargs)

        ax.set_xlabel('iteration index')
        ax.set_ylabel(param_names[idx])
        if use_problem_bounds:
            ax.set_ylim([theta_lb[idx], theta_ub[idx]])

        ax.set_xlim([burn_in, result.sample_result.n_fval + 2])

    if i_chain > 1:
        fig.suptitle('Temperature chain: ' + str(i_chain))
    fig.tight_layout()
    sns.despine()

    return ax


def sampling_scatter(
        result: Result,
        i_chain: int = 0,
        burn_in: int = None,
        stepsize: int = 1,
        size: Tuple[float, float] = None):
    """Parameter scatter plot.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    burn_in:
        Index after burn-in phase, thus also the burn-in length.
    stepsize:
        Only one in `stepsize` values is plotted.
    size:
        Figure size in inches.

    Returns
    -------
    ax:
        The plot axes.
    """
    # TODO: get burn_in from results object
    if burn_in is None:
        burn_in = 0

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = get_data_to_plot(
        result=result, i_chain=i_chain, burn_in=burn_in, stepsize=stepsize)

    sns.set(style="ticks")

    ax = sns.pairplot(
        params_fval.drop(['logPosterior', 'iteration'], axis=1))

    if size is not None:
        ax.fig.set_size_inches(size)

    if i_chain > 1:
        ax.fig.suptitle(f'Temperature chain: {i_chain}')

    return ax


def sampling_1d_marginals(
        result: Result,
        i_chain: int = 0,
        burn_in: int = None,
        stepsize: int = 1,
        plot_type: str = 'both',
        bw='scott',
        size=None):
    """
    Plot marginals.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot. Default: First chain.
    burn_in:
        Index after burn-in phase, thus also the burn-in length.
    stepsize:
        Only one in `stepsize` values is plotted.
    plot_type: {'hist'|'kde'|'both'}
        Specify whether to plot a histogram ('hist'), a kernel density estimate
        ('kde'), or both ('both').
    bw: {'scott', 'silverman' | scalar | pair of scalars}
        Kernel bandwidth method.
    size:
        Figure size in inches.

    Return
    --------
    ax: matplotlib-axes
    """
    # TODO: get burn_in from results object
    if burn_in is None:
        burn_in = 0

    # get data which should be plotted
    nr_params, params_fval, theta_lb, theta_ub = get_data_to_plot(
        result=result, i_chain=i_chain, burn_in=burn_in, stepsize=stepsize)
    param_names = params_fval.columns.values[0:nr_params]

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

    if i_chain > 1:
        fig.suptitle(f'Temperature chain: {i_chain}')
    fig.tight_layout()

    return ax


def get_data_to_plot(
        result: Result, i_chain: int, burn_in: int, stepsize: int):
    """Get the data which should be plotted as a pandas.DataFrame.

    Parameters
    ----------
    result:
        The pyPESTO result object with filled sample result.
    i_chain:
        Which chain to plot.
    burn_in:
        Index after burn-in phase, thus also the burn-in length.
    stepsize:
        Only one in `stepsize` values is plotted.
    """
    # get parameters and fval results as numpy arrays
    arr_param = np.array(result.sample_result['trace_x'][i_chain])

    sample_result: McmcPtResult = result.sample_result

    # thin out by stepsize, from the index burn_in until end of vector
    arr_param = arr_param[np.arange(burn_in, len(arr_param), stepsize)]

    arr_fval = np.array(sample_result.trace_fval[i_chain])
    indices = np.arange(burn_in, len(arr_fval), stepsize)
    arr_fval = arr_fval[indices]
    theta_lb = result.problem.lb
    theta_ub = result.problem.ub

    param_names = result.problem.x_names

    # transform ndarray to pandas for the use of seaborn
    pd_params = pd.DataFrame(arr_param, columns=param_names)
    pd_fval = pd.DataFrame(data=arr_fval, columns=['logPosterior'])

    pd_iter = pd.DataFrame(data=indices, columns=['iteration'])
    params_fval = pd.concat(
        [pd_params, pd_fval, pd_iter], axis=1, ignore_index=False)

    # some global parameters
    nr_params = arr_param.shape[1]  # number of parameters

    return nr_params, params_fval, theta_lb, theta_ub
