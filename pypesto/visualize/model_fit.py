"""
Visualization of the model fit after optimization.

Currently only for PEtab problems.
"""
import numpy as np
import matplotlib.pyplot as plt
import amici.petab_import as petab_import
import petab
from ..problem import Problem
from amici.petab_simulate import simulate_petab
from petab.visualize import plot_problem
from typing import Union, Sequence
from ..result import Result
from amici.plotting import plotObservableTrajectories, plotStateTrajectories
from amici.petab_objective import rdatas_to_simulation_df


def visualize_optimized_model_fit(petab_problem: petab.Problem,
                                  result: Union[Result, Sequence[Result]],
                                  **kwargs
                                  ):
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

    Returns
    -------
    ax: Axis object of the created plot.
    None: In case subplots are saved to file
    """
    if petab_problem is not None:
        if petab is None:
            raise

    problem_parameters = \
        dict(zip(petab_problem.parameter_df.index,
                 result.optimize_result.list[0]['x']))

    amici_model = petab_import.import_petab_problem(
        petab_problem,
        model_output_dir=kwargs.pop('model_output_dir', None),
        force_compile=kwargs.pop('force_compile', False))

    res = simulate_petab(petab_problem, amici_model=amici_model,
                         scaled_parameters=True,
                         problem_parameters=problem_parameters,
                         solver=kwargs.pop('amici_solver', None))

    sim_df = rdatas_to_simulation_df(res["rdatas"],
                                     amici_model,
                                     petab_problem.measurement_df)

    # function to call, to plot data and simulations
    ax = plot_problem(petab_problem=petab_problem,
                      simulations_df=sim_df,
                      **kwargs
                      )
    return ax


def time_trajectory_model(
        result: Union[Result, Sequence[Result]],
        problem: Problem = None,
        # TODO: conditions: Union[str, Sequence[str]] = None,
        timepoints: Union[np.ndarray, Sequence[np.ndarray]] = None,
        n_timepoints: int = 1000,
        state_ids: Union[str, Sequence[str]] = None,
        observable_ids: Union[str, Sequence[str]] = None,):
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
    state_ids:
        Ids of the states to be plotted.
    observable_ids:
        Ids of the observable Ids to be plotted.

    Returns
    -------
    ax: Axis object of the created plot.
    """

    if problem is None:
        problem = result.problem
    # add timepoints as needed
    if timepoints is None:
        end_time = max(problem.objective.edatas[0].getTimepoints())
        timepoints = np.linspace(start=0,
                                 stop=end_time,
                                 num=n_timepoints)
    obj = problem.objective.set_custom_timepoints(
        timepoints_global=timepoints)

    # evaluate objective with return dic = True to get data
    parameters = result.optimize_result.list[0]['x']
    # reduce vector to only include free indices. Needed downstream.
    parameters = problem.get_reduced_vector(parameters)
    ret = obj(parameters, mode='mode_fun',
              sensi_orders=(0,), return_dict=True)

    # if state_ or observable_id's is not None, get indices for these
    state_indices = None
    observable_indices = None
    if state_ids is not None:
        state_indices = [
            problem.objective.amici_model.getStateIds().index(state_id)
            for state_id in state_ids]
    if observable_ids is not None:
        observable_indices = [
            problem.objective.amici_model.getObservableIds().index(obs_id)
            for obs_id in observable_ids]

    fig, ax = plt.subplots(len(ret['rdatas']), 2)
    # enforce two dimensions in case there is only one condition
    ax = np.atleast_2d(ax)

    for i_cond, rdata in enumerate(ret['rdatas']):
        plotStateTrajectories(rdata=rdata,
                              state_indices=state_indices,
                              ax=ax[i_cond, 0],
                              model=problem.objective.amici_model)
        plotObservableTrajectories(rdata=rdata,
                                   observable_indices=observable_indices,
                                   ax=ax[i_cond, 1],
                                   model=problem.objective.amici_model)
    return ax
