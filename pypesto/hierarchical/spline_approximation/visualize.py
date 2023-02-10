from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ...result import Result
from .problem import SplineInnerProblem
from .solver import SplineInnerSolver, get_spline_mapped_simulations


def plot_from_pypesto_result(pypesto_result: Result, **kwargs):
    """Plot the inner solutions from a pypesto result."""

    inner_problem = pypesto_result.problem.objective.calculator.inner_problem

    groups = inner_problem.groups.keys()

    # Get the inner results
    results = []
    for group in groups:
        result = {}
        parameter_values = [
            x.value for x in inner_problem.get_xs_for_group(group)
        ]
        parameter_values.insert(0, 0)

        # Get the reformulated parameters as the difference between the parameter_values
        reformulated_parameter_values = np.diff(parameter_values)
        result['x'] = reformulated_parameter_values
        results.append(result)

    return plot_from_inner_result(inner_problem, results, **kwargs)


def plot_from_inner_result(
    inner_problem: SplineInnerProblem, results: List[Dict], **kwargs
):
    """Plot the inner solutions."""

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
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10), **kwargs)

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

    return fig, axs
