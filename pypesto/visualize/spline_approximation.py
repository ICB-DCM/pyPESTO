import warnings
from typing import Dict, List, Optional, Sequence, Union

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

from ..C import AMICI_SIGMAY, AMICI_Y, CURRENT_SIMULATION, DATAPOINTS, SCIPY_X
from ..problem import Problem
from ..result import Result

try:
    import amici

    from ..hierarchical.spline_approximation.calculator import (
        SplineAmiciCalculator,
    )
    from ..hierarchical.spline_approximation.problem import SplineInnerProblem
    from ..hierarchical.spline_approximation.solver import (
        SplineInnerSolver,
        get_spline_mapped_simulations,
    )
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

    x_dct.update(
        pypesto_result.problem.objective.calculator.noise_dummy_values
    )

    # Get the needed objects from the pypesto problem.
    edatas = pypesto_result.problem.objective.edatas
    parameter_mapping = pypesto_result.problem.objective.parameter_mapping
    amici_model = pypesto_result.problem.objective.amici_model
    amici_solver = pypesto_result.problem.objective.amici_solver
    n_threads = pypesto_result.problem.objective.n_threads
    observable_ids = amici_model.getObservableIds()

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
    sim = [rdata[AMICI_Y] for rdata in inner_rdatas]
    sigma = [rdata[AMICI_SIGMAY] for rdata in inner_rdatas]

    spline_calculator = None
    for (
        calculator
    ) in pypesto_result.problem.objective.calculator.inner_calculators:
        if isinstance(calculator, SplineAmiciCalculator):
            spline_calculator = calculator
            break

    # Get the inner solver and problem.
    inner_solver = spline_calculator.inner_solver
    inner_problem = spline_calculator.inner_problem

    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    return plot_splines_from_inner_result(
        inner_problem, inner_results, observable_ids, **kwargs
    )


def plot_splines_from_inner_result(
    inner_problem: 'SplineInnerProblem',
    results: List[Dict],
    observable_ids=None,
    **kwargs,
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

        s = result[SCIPY_X]

        inner_parameters = np.array([x.value for x in xs])
        measurements = inner_problem.groups[group][DATAPOINTS]
        simulation = inner_problem.groups[group][CURRENT_SIMULATION]

        # For the simulation, get the spline bases
        delta_c, spline_bases, n = SplineInnerSolver._rescale_spline_bases(
            self=None,
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
        axs[group_idx].plot(
            spline_bases, inner_parameters, 'g.', label='Spline knots'
        )
        axs[group_idx].plot(
            spline_bases,
            inner_parameters,
            linestyle='-',
            color='g',
            label='Spline function',
        )
        axs[group_idx].plot(
            simulation, mapped_simulations, 'r^', label='Mapped simulation'
        )
        axs[group_idx].legend()
        if observable_ids is not None:
            axs[group_idx].set_title(f'{observable_ids[group-1]}')
        else:
            axs[group_idx].set_title(f'Group {group}')

        axs[group_idx].set_xlabel('Model output')
        axs[group_idx].set_ylabel('Measurements')

    for ax in axs[len(results) :]:
        ax.remove()

    return fig, axs


def _add_spline_mapped_simulations_to_model_fit(
    result: Union[Result, Sequence[Result]],
    pypesto_problem: Problem,
    start_index: int = 0,
    axes: Optional[plt.Axes] = None,
) -> Union[matplotlib.axes.Axes, None]:
    """Visualize the spline optimized model fit.

    Adds the spline-mapped simulation to the axes given by
    `pypesto.visualize.model_fit.visualize_optimized_model_fit`.
    For further details on documentation see
    :py:func:`pypesto.visualize.model_fit.visualize_optimized_model_fit`.
    """

    # If no visualize_optimized_model_fit axes were given,
    # return None.
    if axes is None:
        return None

    # Get the parameters from the pypesto result for the start_index.
    x_dct = dict(
        zip(
            pypesto_problem.objective.x_ids,
            result.optimize_result.list[start_index]['x'],
        )
    )
    x_dct.update(pypesto_problem.objective.calculator.noise_dummy_values)
    # Get the needed objects from the pypesto problem.
    edatas = pypesto_problem.objective.edatas
    parameter_mapping = pypesto_problem.objective.parameter_mapping
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    n_threads = pypesto_problem.objective.n_threads

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
    sim = [rdata[AMICI_Y] for rdata in inner_rdatas]
    sigma = [rdata[AMICI_SIGMAY] for rdata in inner_rdatas]

    spline_calculator = None
    for calculator in pypesto_problem.objective.calculator.inner_calculators:
        if isinstance(calculator, SplineAmiciCalculator):
            spline_calculator = calculator
            break

    # Get the inner solver and problem.
    inner_solver = spline_calculator.inner_solver
    inner_problem = spline_calculator.inner_problem

    # Solve the inner problem.
    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    # Get the observable ids.
    observable_ids = amici_model.getObservableIds()

    for inner_result, group in zip(inner_results, inner_problem.groups):
        observable_id = observable_ids[group - 1]
        # Get the ax for the current observable.
        ax = [
            ax
            for ax in axes.values()
            if observable_id in ax.legend().get_texts()[0].get_text().split()
        ][0]

        # Get the inner parameters and simulation.
        xs = inner_problem.get_xs_for_group(group)
        s = inner_result[SCIPY_X]

        inner_parameters = np.array([x.value for x in xs])
        simulation = inner_problem.groups[group][CURRENT_SIMULATION]

        # For the simulation, get the spline bases
        delta_c, spline_bases, n = SplineInnerSolver._rescale_spline_bases(
            self=None,
            sim_all=simulation,
            N=len(inner_parameters),
            K=len(simulation),
        )
        # and the spline-mapped simulations.
        mapped_simulations = get_spline_mapped_simulations(
            s, simulation, len(inner_parameters), delta_c, spline_bases, n
        )

        # Plot the spline-mapped simulations to the ax with same color
        # and timepoints as the lines which have 'simulation' in their label.
        plotted_index = 0
        for line in ax.lines:
            if 'simulation' in line.get_label():
                color = line.get_color()
                timepoints = line.get_xdata()
                condition_mapped_simulations = mapped_simulations[
                    plotted_index : len(timepoints) + plotted_index
                ]
                plotted_index += len(timepoints)

                ax.plot(
                    timepoints,
                    condition_mapped_simulations,
                    color=color,
                    linestyle='dotted',
                    marker='^',
                )

        # Add linestyle='dotted' and marker='^' to the legend as black spline mapped simulations.
        ax.plot(
            [],
            [],
            color='black',
            linestyle='dotted',
            marker='^',
            label='Spline mapped simulation',
        )

    # Reset the legend.
    for ax in axes.values():
        ax.legend()

    return axes
