import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ...result import Result
from .problem import SplineInnerProblem
from .solver import SplineInnerSolver, get_spline_mapped_simulations

try:
    import amici
except ImportError:
    pass


def plot_splines_from_pypesto_result(
    pypesto_result: Result, start_index=0, **kwargs
):
    """Plot the inner solutions from a pypesto result.

    Parameters
    ----------
    pypesto_result:
        The pypesto result.
    start_index:
        The index of the pypesto_result.optimize_result.list to plot.
    kwargs:
        Additional arguments to pass to the plotting function.

    Returns
    -------
    fig:
        The figure.
    ax:
        The axes.
    """

    # Get the parameters from the pypesto result for the start_index.
    x_dct = dict(
        zip(
            pypesto_result.problem.objective.x_ids,
            pypesto_result.optimize_result.list[start_index]['x'],
        )
    )

    # Get the needed objects from the pypesto problem.
    edatas = pypesto_result.problem.objective.edatas
    parameter_mapping = pypesto_result.problem.objective.parameter_mapping
    amici_model = pypesto_result.problem.objective.amici_model
    amici_solver = pypesto_result.problem.objective.amici_solver
    n_threads = pypesto_result.problem.objective.n_threads

    # Fill in the parameters.
    amici.parameter_mapping.fill_in_parameters(
        edatas=edatas,
        problem_parameters=x_dct,
        scaled_parameters=True,
        parameter_mapping=parameter_mapping,
        amici_model=amici_model,
    )

    # Simulate the model with the parameters from the pypesto result.
    inner_rdatas = amici.runAmiciSimulations(
        amici_model,
        amici_solver,
        edatas,
        num_threads=min(n_threads, len(edatas)),
    )

    # If any amici simulation failed, raise warning and return None.
    if any(rdata.status != amici.AMICI_SUCCESS for rdata in inner_rdatas):
        warnings.warn(
            'Warning: Some AMICI simulations failed. Cannot plot inner solutions.'
        )
        return None

    # Get simulation and sigma.
    sim = [rdata['y'] for rdata in inner_rdatas]
    sigma = [rdata['sigmay'] for rdata in inner_rdatas]

    # Get the inner solver and problem.
    inner_solver = pypesto_result.problem.objective.calculator.inner_solver
    inner_problem = pypesto_result.problem.objective.calculator.inner_problem

    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    return plot_splines_from_inner_result(
        inner_problem, inner_results, **kwargs
    )


def plot_splines_from_inner_result(
    inner_problem: SplineInnerProblem, results: List[Dict], **kwargs
):
    """Plot the inner solutions.

    Parameters
    ----------
    inner_problem:
        The inner problem.
    results:
        The results from the inner solver.
    kwargs:
        Additional arguments to pass to the plotting function.

    Returns
    -------
    fig:
        The figure.
    ax:
        The axes.
    """

    if len(results) != len(inner_problem.groups):
        raise ValueError(
            "Number of results must be equal to number of groups of the inner subproblem."
        )

    # Get the number of groups
    n_groups = len(inner_problem.groups)

    if n_groups == 1:
        # Make figure with only one plot
        fig, ax = plt.subplots(1, 1, **kwargs)

        axs = [ax]
    else:
        # Choose number of rows and columns to be used for the subplots
        n_rows = int(np.ceil(np.sqrt(n_groups)))
        n_cols = int(np.ceil(n_groups / n_rows))

        # Make as many subplots as there are groups
        fig, axs = plt.subplots(n_rows, n_cols, **kwargs)

        # Flatten the axes array
        axs = axs.flatten()

    # for each result and group, plot the inner solution
    for result, group in zip(results, inner_problem.groups):
        group_idx = list(inner_problem.groups.keys()).index(group)

        # For each group get the inner parameters and simulation
        xs = inner_problem.get_xs_for_group(group)

        s = result['x']

        inner_parameters = np.array([x.value for x in xs])
        measurements = inner_problem.groups[group]['datapoints']
        simulation = inner_problem.groups[group]['current_simulation']

        # For the simulation, get the spline bases
        delta_c, spline_bases, n = SplineInnerSolver._rescale_spline_bases(
            self='a',
            sim_all=simulation,
            N=len(inner_parameters),
            K=len(simulation),
        )
        mapped_simulations = get_spline_mapped_simulations(
            s, simulation, len(inner_parameters), delta_c, spline_bases, n
        )

        axs[group_idx].plot(
            simulation, measurements, 'bs', label='Measurements'
        )
        axs[group_idx].plot(spline_bases, inner_parameters, 'g')
        axs[group_idx].plot(
            spline_bases, inner_parameters, 'g.', label='Spline parameters'
        )
        axs[group_idx].plot(
            simulation, mapped_simulations, 'r^', label='Mapped simulation'
        )
        axs[group_idx].legend()
        axs[group_idx].set_title(f'Group {group}')

        axs[group_idx].set_xlabel('Model output')
        axs[group_idx].set_ylabel('Measurements')

    for ax in axs[len(results) :]:
        ax.remove()

    return fig, axs
