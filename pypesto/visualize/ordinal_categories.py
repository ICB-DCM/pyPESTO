import warnings
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

try:
    import amici
except ImportError:
    pass

from ..hierarchical.optimal_scaling.parameter import OptimalScalingParameter
from ..hierarchical.optimal_scaling.problem import OptimalScalingProblem
from ..hierarchical.optimal_scaling.solver import (
    OptimalScalingInnerSolver,
    compute_interval_constraints,
    get_bounds_for_category,
    undo_inner_parameter_reparameterization,
)
from ..result import Result


def plot_categories_from_pypesto_result(
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
        Additional arguments to pass to the figure.

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
    petab_problem = (
        pypesto_result.problem.objective.amici_object_builder.petab_problem
    )
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
    timepoints = [rdata['ts'] for rdata in inner_rdatas]
    condition_ids = [edata.id for edata in edatas]
    condition_ids_from_petab = list(petab_problem.condition_df.index)
    petab_condition_ordering = [
        condition_ids.index(condition_id)
        for condition_id in condition_ids_from_petab
    ]

    # Get the inner solver and problem.
    inner_solver = pypesto_result.problem.objective.calculator.inner_solver
    inner_problem = pypesto_result.problem.objective.calculator.inner_problem

    inner_results = inner_solver.solve(inner_problem, sim)

    return plot_categories_from_inner_result(
        inner_problem,
        inner_solver,
        inner_results,
        sim,
        timepoints,
        condition_ids_from_petab,
        petab_condition_ordering,
        **kwargs,
    )


def plot_categories_from_inner_result(
    inner_problem: OptimalScalingProblem,
    inner_solver: OptimalScalingInnerSolver,
    results: List[Dict],
    simulation: List[np.ndarray],
    timepoints: List[np.ndarray],
    condition_ids: List[str] = None,
    petab_condition_ordering: List[str] = None,
    **kwargs,
):
    """Plot the inner solutions.

    Parameters
    ----------
    inner_problem:
        The inner problem.
    inner_solver:
        The inner solver.
    results:
        The results from the inner solver.
    simulation:
        The model simulation.
    timepoints:
        The timepoints of the simulation.
    kwargs:
        Additional arguments to pass to the figure.

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
    options = inner_solver.options

    # If there is only one group, make a figure with only one plot
    if n_groups == 1:
        # Make figure with only one plot
        fig, ax = plt.subplots(1, 1, **kwargs)

        axs = [ax]
    # If there are multiple groups, make a figure with multiple plots
    else:
        # Choose number of rows and columns to be used for the subplots
        n_rows = int(np.ceil(np.sqrt(n_groups)))
        n_cols = int(np.ceil(n_groups / n_rows))

        # Make as many subplots as there are groups
        fig, axs = plt.subplots(n_rows, n_cols, **kwargs)

        # Increase the spacing between the subplots
        fig.subplots_adjust(hspace=0.35, wspace=0.25)

        # Flatten the axes array
        axs = axs.flatten()

    # for each result and group, plot the inner solution
    for result, group in zip(results, inner_problem.groups):
        group_idx = list(inner_problem.groups.keys()).index(group)

        # For each group get the inner parameters and simulation
        xs = inner_problem.get_cat_ub_parameters_for_group(group)

        interval_range, interval_gap = compute_interval_constraints(
            xs, simulation, options
        )

        # Get surrogate datapoints and category bounds
        (
            simulation_all,
            surrogate_all,
            timepoints_all,
            upper_bounds_all,
            lower_bounds_all,
        ) = _get_data_for_plotting(
            xs,
            result['x'],
            simulation,
            timepoints,
            interval_range,
            interval_gap,
            options,
        )

        # Get the number of distinct timepoints in timepoints_all
        # where timepoints_all is a list of numpy arrays of timepoints
        n_distinct_timepoints = len(np.unique(np.concatenate(timepoints_all)))

        # If there is only one distinct timepoint, plot with respect to conditions
        if n_distinct_timepoints == 1:
            # Merge the simulation, surrogate, and bounds across conditions
            simulation_all = np.concatenate(simulation_all)
            surrogate_all = np.concatenate(surrogate_all)
            upper_bounds_all = np.concatenate(upper_bounds_all)
            lower_bounds_all = np.concatenate(lower_bounds_all)

            # Change ordering of simulation, surrogate data and bounds to petab condition ordering
            simulation_all = simulation_all[petab_condition_ordering]
            surrogate_all = surrogate_all[petab_condition_ordering]
            upper_bounds_all = upper_bounds_all[petab_condition_ordering]
            lower_bounds_all = lower_bounds_all[petab_condition_ordering]

            # Plot the categories and surrogate data across conditions

            axs[group_idx].plot(
                condition_ids,
                simulation_all,
                'b.',
                label='Simulation',
            )
            axs[group_idx].plot(
                condition_ids,
                simulation_all,
                'b',
                label='Simulation',
            )
            axs[group_idx].plot(
                condition_ids,
                surrogate_all,
                'rx',
                label='Surrogate data',
            )

            _plot_category_rectangles(
                axs[group_idx],
                condition_ids,
                upper_bounds_all,
                lower_bounds_all,
            )

            # Set the condition xticks on an angle
            axs[group_idx].tick_params(axis='x', rotation=25)
            axs[group_idx].legend()
            axs[group_idx].set_title(f'Group {group}')

            axs[group_idx].set_xlabel('Conditions')
            axs[group_idx].set_ylabel('Simulation/Surrogate data')

        # Plotting across timepoints
        else:
            n_conditions = len(simulation_all)

            # If there is only one condition, we don't need separate colors
            # for the different conditions
            if n_conditions == 1:
                axs[group_idx].plot(
                    timepoints_all[0],
                    simulation_all[0],
                    'b.',
                    label='Simulation',
                )
                axs[group_idx].plot(
                    timepoints_all[0],
                    simulation_all[0],
                    'b',
                    label='Simulation',
                )
                axs[group_idx].plot(
                    timepoints_all[0],
                    surrogate_all[0],
                    'rx',
                    label='Surrogate data',
                )

                # Plot the categorie rectangles
                _plot_category_rectangles(
                    axs[group_idx],
                    timepoints_all[0],
                    upper_bounds_all[0],
                    lower_bounds_all[0],
                )

            # If there are multiple conditions, we need separate colors
            # for the different conditions
            else:
                # Get as many colors as there are conditions
                colors = plt.cm.rainbow(np.linspace(0, 1, len(simulation_all)))

                # Plot the categories and surrogate data for all conditions.
                for condition_index, condition_id, color in zip(
                    range(len(simulation_all)), condition_ids, colors
                ):
                    # Plot the categories and surrogate data for the current condition
                    axs[group_idx].plot(
                        timepoints_all[condition_index],
                        simulation_all[condition_index],
                        linestyle='-',
                        marker='.',
                        color=color,
                        label=condition_id,
                    )
                    axs[group_idx].plot(
                        timepoints_all[condition_index],
                        surrogate_all[condition_index],
                        'x',
                        color=color,
                    )

                # Get all unique timepoints in ascending order
                unique_timepoints = np.unique(np.concatenate(timepoints_all))

                # Gather timepoints for each category in a dictionary
                # with upper, lower bound tuple as key and list of timepoints as value
                category_timepoints_dict = {}

                for condition_idx in range(len(simulation_all)):
                    for upper_bound, lower_bound, timepoint in zip(
                        upper_bounds_all[condition_idx],
                        lower_bounds_all[condition_idx],
                        timepoints_all[condition_idx],
                    ):
                        if (
                            upper_bound,
                            lower_bound,
                        ) not in category_timepoints_dict:
                            category_timepoints_dict[
                                (upper_bound, lower_bound)
                            ] = [timepoint]
                        else:
                            category_timepoints_dict[
                                (upper_bound, lower_bound)
                            ].append(timepoint)

                # Plot the category rectangles
                _plot_category_rectangles_across_conditions(
                    axs[group_idx],
                    category_timepoints_dict,
                    unique_timepoints,
                )

                # Add to legend meaning of x, and -o- markers.
                axs[group_idx].plot(
                    [],
                    [],
                    'x',
                    color='black',
                    label='Surrogate data',
                )
                axs[group_idx].plot(
                    [],
                    [],
                    linestyle='-',
                    marker='.',
                    color='black',
                    label='Simulation',
                )

            axs[group_idx].legend()
            axs[group_idx].set_title(f'Group {group}')

            axs[group_idx].set_xlabel('Timepoints')
            axs[group_idx].set_ylabel('Simulation/Surrogate data')

    for ax in axs[len(results) :]:
        ax.remove()

    return fig, axs


def _plot_category_rectangles_across_conditions(
    ax, category_timepoints_dict, unique_timepoints
) -> None:
    for (
        upper_bound,
        lower_bound,
    ), timepoints in category_timepoints_dict.items():
        # If the largest timepoint is not the last unique timepoint, add the next unique timepoint
        # to the list of timepoints
        max_timepoint_unique_ind = np.where(
            unique_timepoints == max(timepoints)
        )[0][0]
        if max_timepoint_unique_ind + 1 < len(unique_timepoints):
            timepoints.append(unique_timepoints[max_timepoint_unique_ind + 1])

        # Plot the category rectangle
        ax.fill_between(
            timepoints,
            [upper_bound] * len(timepoints),
            [lower_bound] * len(timepoints),
            color='gray',
            alpha=0.5,
        )

    # Add to legend meaning of gray rectangles.
    ax.fill_between(
        [],
        [],
        [],
        color='gray',
        alpha=0.5,
        label='Categories',
    )


def _plot_category_rectangles(
    ax, timepoints, upper_bounds, lower_bounds
) -> None:
    """Plot the category rectangles."""
    interval_length = 0
    for i in range(len(timepoints)):
        if i + 1 == len(timepoints) or upper_bounds[i + 1] != upper_bounds[i]:
            if i + 1 == len(timepoints):
                ax.fill_between(
                    timepoints[i - interval_length : i + 1],
                    upper_bounds[i - interval_length : i + 1],
                    lower_bounds[i - interval_length : i + 1],
                    color='gray',
                    alpha=0.5,
                )
            else:
                ax.fill_between(
                    timepoints[i - interval_length : i + 2],
                    np.concatenate(
                        (
                            upper_bounds[i - interval_length : i + 1],
                            [upper_bounds[i]],
                        )
                    ),
                    np.concatenate(
                        (
                            lower_bounds[i - interval_length : i + 1],
                            [lower_bounds[i]],
                        )
                    ),
                    color='gray',
                    alpha=0.5,
                )
            interval_length = 0
        else:
            interval_length += 1
    # Add to legend meaning of rectangles
    ax.fill_between(
        [],
        [],
        [],
        color='gray',
        alpha=0.5,
        label='Categories',
    )


def _get_data_for_plotting(
    inner_parameters: List[OptimalScalingParameter],
    optimal_scaling_bounds: List,
    sim: List[np.ndarray],
    timepoints: List[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: Dict,
):
    """Return data in the form suited for plotting."""
    if options['reparameterized']:
        optimal_scaling_bounds = undo_inner_parameter_reparameterization(
            optimal_scaling_bounds,
            inner_parameters,
            interval_gap,
            interval_range,
        )

    simulation_all = []
    surrogate_all = []
    timepoints_all = []
    upper_bounds_all = []
    lower_bounds_all = []

    for condition_index in range(len(sim)):
        cond_simulation = []
        cond_surrogate = []
        cond_timepoints = []
        cond_upper_bounds = []
        cond_lower_bounds = []

        for inner_parameter in inner_parameters:
            upper_bound, lower_bound = get_bounds_for_category(
                inner_parameter, optimal_scaling_bounds, interval_gap, options
            )
            # Get the condition specific simulation, mask, and timepoints
            sim_i = sim[condition_index]
            mask_i = inner_parameter.ixs[condition_index]
            t_i = timepoints[condition_index]

            y_sim = sim_i[mask_i]

            # If there is no measurement in this
            # condition for this category, skip it
            if len(y_sim) == 0:
                continue

            if mask_i.ndim == 1:
                t_sim = t_i[mask_i]
            else:
                observable_index = [
                    i for i in range(len(mask_i.T)) if any(mask_i.T[i])
                ][0]
                t_sim = timepoints[condition_index][mask_i.T[observable_index]]

            for y_sim_i in y_sim:
                if lower_bound > y_sim_i:
                    y_surrogate = lower_bound
                elif y_sim_i > upper_bound:
                    y_surrogate = upper_bound
                elif lower_bound <= y_sim_i <= upper_bound:
                    y_surrogate = y_sim_i
                else:
                    continue
                cond_surrogate.append(y_surrogate)
                cond_upper_bounds.append(upper_bound)
                cond_lower_bounds.append(lower_bound)
            cond_simulation.extend(y_sim)
            cond_timepoints.extend(t_sim)

        # Sort the surrogate datapoints and categories by timepoints, ascending.
        cond_simulation = np.array(cond_simulation)
        cond_surrogate = np.array(cond_surrogate)
        cond_timepoints = np.array(cond_timepoints)
        cond_upper_bounds = np.array(cond_upper_bounds)
        cond_lower_bounds = np.array(cond_lower_bounds)
        sort_idx = np.argsort(cond_timepoints)

        cond_simulation = cond_simulation[sort_idx]
        cond_surrogate = cond_surrogate[sort_idx]
        cond_timepoints = cond_timepoints[sort_idx]
        cond_upper_bounds = cond_upper_bounds[sort_idx]
        cond_lower_bounds = cond_lower_bounds[sort_idx]

        # Add the condition surrogate datapoints and categories to the list of all conditions.
        simulation_all.append(cond_simulation)
        surrogate_all.append(cond_surrogate)
        timepoints_all.append(cond_timepoints)
        upper_bounds_all.append(cond_upper_bounds)
        lower_bounds_all.append(cond_lower_bounds)

    return (
        simulation_all,
        surrogate_all,
        timepoints_all,
        upper_bounds_all,
        lower_bounds_all,
    )
