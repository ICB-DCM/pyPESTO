import logging
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Sequence, Tuple

from ..result import Result
from ..sample import McmcPtResult

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
        params_fval.drop(['logPosterior', 'iteration'], axis=1))

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
