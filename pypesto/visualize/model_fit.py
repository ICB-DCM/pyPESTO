"""
Visualization of the model fit after optimization.

Currently only for PEtab problems.
"""
from typing import Sequence, Union

import amici
import amici.plotting
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import petab
from amici.petab_objective import rdatas_to_simulation_df
from petab.visualize import plot_problem

from ..C import CENSORED, NONLINEAR_MONOTONE, ORDINAL, RDATAS
from ..petab.importer import get_petab_non_quantitative_data_types
from ..problem import Problem
from ..result import Result
from .ordinal_categories import plot_categories_from_pypesto_result
from .spline_approximation import _add_spline_mapped_simulations_to_model_fit

AmiciModel = Union['amici.Model', 'amici.ModelPtr']

__all__ = ["visualize_optimized_model_fit", "time_trajectory_model"]


def visualize_optimized_model_fit(
    petab_problem: petab.Problem,
    result: Union[Result, Sequence[Result]],
    pypesto_problem: Problem,
    start_index: int = 0,
    return_dict: bool = False,
    unflattened_petab_problem: petab.Problem = None,
    **kwargs,
) -> Union[matplotlib.axes.Axes, None]:
    """
    Visualize the optimized model fit of a PEtab problem.

    Function calls the PEtab visualization file of the ``petab_problem`` and
    visualizes the fit of the optimized parameter. Common additional
    argument is ``subplot_dir`` to specify the directory each subplot is
    saved to. Further keyword arguments are delegated to
    :func:`petab.visualize.plot_with_vis_spec`, see there for more information.

    Parameters
    ----------
    petab_problem:
        The :py:class:`petab.Problem` that was optimized.
    result:
        The result object from optimization.
    start_index:
        The index of the optimization run in `result.optimize_result.list`.
        Ignored if `problem_parameters` is provided.
    pypesto_problem:
        The pyPESTO problem.
    return_dict:
        Return plot and simulation results as a dictionary.
    unflattened_petab_problem:
        If the original PEtab problem is flattened, this can be passed
        to plot with the original unflattened problem.
    kwargs:
        Passed to :func:`petab.visualize.plot_problem`.

    Returns
    -------
    axes: `matplotlib.axes.Axes` object of the created plot.
    None: In case subplots are saved to file
    """
    x = result.optimize_result.list[start_index]['x'][
        pypesto_problem.x_free_indices
    ]
    objective_result = pypesto_problem.objective(x, return_dict=True)

    simulation_df = rdatas_to_simulation_df(
        objective_result[RDATAS],
        pypesto_problem.objective.amici_model,
        petab_problem.measurement_df,
    )

    # handle flattened PEtab problems
    petab_problem_to_plot = petab_problem
    if unflattened_petab_problem:
        simulation_df = petab.core.unflatten_simulation_df(
            simulation_df=simulation_df,
            petab_problem=unflattened_petab_problem,
        )
        petab_problem_to_plot = unflattened_petab_problem

    # plot
    axes = plot_problem(
        petab_problem=petab_problem_to_plot,
        simulations_df=simulation_df,
        **kwargs,
    )

    non_quantitative_data_types = get_petab_non_quantitative_data_types(
        petab_problem
    )

    if non_quantitative_data_types:
        if (
            ORDINAL in non_quantitative_data_types
            or CENSORED in non_quantitative_data_types
        ):
            axes = plot_categories_from_pypesto_result(
                result,
                start_index=start_index,
                axes=axes,
            )
        if NONLINEAR_MONOTONE in non_quantitative_data_types:
            axes = _add_spline_mapped_simulations_to_model_fit(
                result=result,
                pypesto_problem=pypesto_problem,
                start_index=start_index,
                axes=axes,
            )

    if return_dict:
        return {
            'axes': axes,
            'objective_result': objective_result,
            'simulation_df': simulation_df,
        }
    return axes


def time_trajectory_model(
    result: Union[Result, Sequence[Result]],
    problem: Problem = None,
    # TODO: conditions: Union[str, Sequence[str]] = None,
    timepoints: Union[np.ndarray, Sequence[np.ndarray]] = None,
    n_timepoints: int = 1000,
    start_index: int = 0,
    state_ids: Union[str, Sequence[str]] = None,
    state_names: Union[str, Sequence[str]] = None,
    observable_ids: Union[str, Sequence[str]] = None,
) -> Union[matplotlib.axes.Axes, None]:
    """
    Visualize the time trajectory of the model with given timepoints.

    It does this by calling the amici plotting routines.

    Parameters
    ----------
    result:
        The result object from optimization.
    problem:
        A pypesto problem. Default is 'None' in which case result.problem is
        used. Needed in case the result is loaded from hdf5.
    timepoints:
        Array of timepoints, at which the trajectory will be plotted.
    n_timepoints:
        Number of timepoints to be plotted between 0 and last measurement of
        the model. Only used when timepoints==None.
    start_index:
        Index of Optimization run to be plotted. Default is best start.
    state_ids:
        Ids of the states to be plotted.
    state_names:
        Names of the states to be plotted.
    observable_ids:
        Ids of the observables to be plotted.

    Returns
    -------
    axes:
        `matplotlib.axes.Axes` object of the plot.
    """
    if problem is None:
        problem = result.problem
    # add timepoints as needed
    if timepoints is None:
        end_time = max(problem.objective.edatas[0].getTimepoints())
        timepoints = np.linspace(start=0, stop=end_time, num=n_timepoints)
    obj = problem.objective.set_custom_timepoints(timepoints_global=timepoints)

    # evaluate objective with return dic = True to get data
    parameters = result.optimize_result.list[start_index]['x']
    # reduce vector to only include free indices. Needed downstream.
    parameters = problem.get_reduced_vector(parameters)
    ret = obj(parameters, mode='mode_fun', sensi_orders=(0,), return_dict=True)

    if state_ids == [] and state_names == []:
        axes = _time_trajectory_model_without_states(
            model=problem.objective.amici_model,
            rdatas=ret['rdatas'],
            observable_ids=observable_ids,
        )
    else:
        axes = _time_trajectory_model_with_states(
            model=problem.objective.amici_model,
            rdatas=ret['rdatas'],
            state_ids=state_ids,
            state_names=state_names,
            observable_ids=observable_ids,
        )

    return axes


def _time_trajectory_model_with_states(
    model: AmiciModel,
    rdatas: Union['amici.ReturnData', Sequence['amici.ReturnData']],
    state_ids: Sequence[str],
    state_names: Sequence[str],
    observable_ids: Union[str, Sequence[str]],
):
    """
    Visualizes both, states and observables.

    Helper function for time_trajectory_model.

    Parameters
    ----------
    model:
        The amici.Model from the model of interest. Used to annotate the plot.
    rdatas:
        The data to be plotted. Each entry in the Sequence corresponds to a
        condition.
    state_ids:
        Ids of the states to be plotted.
    state_names:
        Names of the states to be plotted.
    observable_ids:
        Ids of the observable Ids to be plotted.

    Returns
    -------
    axes:
        `matplotlib.axes.Axes` object of the plot.
    """
    # if state_name, state_id or observable_id is not None, get indices
    # for these the AMICI plotting functions default to all indices if
    # `None` is specified.
    state_indices_by_id = []
    state_indices_by_name = []
    if state_ids is not None:
        state_indices_by_id = [
            model.getStateIds().index(state_id) for state_id in state_ids
        ]
    if state_names is not None:
        state_indices_by_name = [
            model.getStateNames().index(state_name)
            for state_name in state_names
        ]
    state_indices = list(set(state_indices_by_id + state_indices_by_name))
    if state_indices == []:
        state_indices = None

    observable_indices = None
    if observable_ids is not None:
        observable_indices = [
            model.getObservableIds().index(obs_id) for obs_id in observable_ids
        ]

    fig, axes = plt.subplots(len(rdatas), 2)
    # enforce two dimensions in case there is only one condition
    axes = np.atleast_2d(axes)

    for i_cond, rdata in enumerate(rdatas):
        amici.plotting.plotStateTrajectories(
            rdata=rdata,
            state_indices=state_indices,
            ax=axes[i_cond, 0],
            model=model,
        )
        amici.plotting.plotObservableTrajectories(
            rdata=rdata,
            observable_indices=observable_indices,
            ax=axes[i_cond, 1],
            model=model,
        )
    return axes


def _time_trajectory_model_without_states(
    model: AmiciModel,
    rdatas: Union['amici.ReturnData', Sequence['amici.ReturnData']],
    observable_ids: Union[str, Sequence[str]],
):
    """
    Visualize both, states and observables.

    Helper function for time_trajectory_model.

    Parameters
    ----------
    model:
        The amici.Model from the model of interest. Used to annotate the plot.
    rdatas:
        The data to be plotted. Each entry in the Sequence corresponds to a
        condition.
    observable_ids:
        Ids of the observables to be plotted.

    Returns
    -------
    axes:
        `matplotlib.axes.Axes` object of the plot.
    """
    # if observable_id's is not None, get indices for these
    # the AMICI plotting functions default to all indices if `None` is
    # specified.
    observable_indices = None
    if observable_ids is not None:
        observable_indices = [
            model.getObservableIds().index(obs_id) for obs_id in observable_ids
        ]

    fig, axes = plt.subplots(len(rdatas))

    for i_cond, rdata in enumerate(rdatas):
        amici.plotting.plotObservableTrajectories(
            rdata=rdata,
            observable_indices=observable_indices,
            ax=axes[i_cond] if len(rdatas) > 1 else axes,
            model=model,
        )
    return axes
