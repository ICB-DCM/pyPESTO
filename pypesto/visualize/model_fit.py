"""
Visualization of the model fit after optimization.

Currently only for PEtab problems.
"""
from typing import Dict, Sequence, Union

import amici
import amici.petab_import as petab_import
import amici.plotting
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import petab
from amici.petab_objective import rdatas_to_simulation_df
from amici.petab_simulate import simulate_petab
from petab.visualize import plot_problem

from ..problem import Problem
from ..result import Result

AmiciModel = Union['amici.Model', 'amici.ModelPtr']


def visualize_optimized_model_fit(
    petab_problem: petab.Problem,
    result: Union[Result, Sequence[Result]] = None,
    start_index: int = 0,
    problem_parameters: Dict[str, float] = None,
    model_output_dir: str = None,
    force_compile: bool = False,
    amici_solver: amici.Solver = None,
    return_dict: bool = False,
    **kwargs,
) -> Union[matplotlib.axes.Axes, None]:
    """
    Visualize the optimized model fit of a PEtab problem.

    Function calls the PEtab visualization file of the petab_problem and
    visualizes the fit of the optimized parameter. Common additional
    argument is `subplot_dir` to specify the directory each subplot is
    saved to. Further keyword arguments are delegated to
    petab.visualize.plot_with_vis_spec, see there for more information.

    Parameters
    ----------
    petab_problem:
        The :py:class:`petab.Problem` that was optimized.
    result:
        The result object from optimization.
    start_index:
        The index of the optimization run in `result.optimize_result.list`.
        Ignored if `problem_parameters` is provided.
    problem_parameters:
        The (scaled) problem parameters to simulate.
        Defaults to using `start_index`.
    model_output_dir, force_compile:
        Passed to `amici.petab_import.import_petab_problem`.
    amici_solver:
        Passed to `amici.petab_objective.simulate_petab` as `solver`.
    return_dict:
        Return plot and simulation results as a dictionary.
    kwargs:
        Passed to `petab.visualize.plot_problem`.

    Returns
    -------
    axes: `matplotlib.axes.Axes` object of the created plot.
    None: In case subplots are saved to file
    """
    if petab_problem is not None:
        if petab is None:
            raise

    if problem_parameters is None:
        if result is None:
            raise ValueError(
                'Please provide a pyPESTO `result` or the '
                '`problem_parameters` directly.'
            )
        problem_parameters = dict(
            zip(
                petab_problem.parameter_df.index,
                result.optimize_result.list[start_index]['x'],
            )
        )

    amici_model = petab_import.import_petab_problem(
        petab_problem,
        model_output_dir=model_output_dir,
        force_compile=force_compile,
    )

    res = simulate_petab(
        petab_problem,
        amici_model=amici_model,
        scaled_parameters=True,
        problem_parameters=problem_parameters,
        solver=amici_solver,
    )

    sim_df = rdatas_to_simulation_df(
        res["rdatas"], amici_model, petab_problem.measurement_df
    )

    # function to call, to plot data and simulations
    axes = plot_problem(
        petab_problem=petab_problem, simulations_df=sim_df, **kwargs
    )
    if return_dict:
        return {
            'axes': axes,
            'amici_result': res,
            'simulation_df': sim_df,
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
            ax=axes[i_cond],
            model=model,
        )
    return axes
