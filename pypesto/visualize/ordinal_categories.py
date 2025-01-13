import warnings
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import pypesto

import matplotlib.pyplot as plt
import numpy as np

try:
    import amici
    from amici.petab.conditions import fill_in_parameters
    from petab.v1.C import OBSERVABLE_ID

    from ..hierarchical.ordinal.calculator import OrdinalCalculator
    from ..hierarchical.ordinal.parameter import OrdinalParameter
    from ..hierarchical.ordinal.solver import (
        compute_interval_constraints,
        get_bounds_for_category,
        undo_inner_parameter_reparameterization,
    )
except ImportError:
    pass


from ..C import (
    AMICI_SIGMAY,
    AMICI_T,
    AMICI_Y,
    CENSORED,
    MEASUREMENT_TYPE,
    ORDINAL,
    QUANTITATIVE_DATA,
    QUANTITATIVE_IXS,
    REPARAMETERIZED,
    SCIPY_X,
)
from ..result import Result


def plot_categories_from_pypesto_result(
    pypesto_result: Result,
    start_index=0,
    axes: Optional[plt.Axes] = None,
    **kwargs,
):
    """Plot the inner solutions from a pypesto result.

    Parameters
    ----------
    pypesto_result:
        The pypesto result.
    start_index:
        The index of the pypesto_result.optimize_result.list to plot.
    axes:
        The optional axes to plot on.
    kwargs:
        Additional arguments to pass to the figure.

    Returns
    -------
    fig:
        The figure.
    axes:
        The axes.
    """

    # Get the parameters from the pypesto result for the start_index.
    x_dct = dict(
        zip(
            pypesto_result.problem.objective.x_ids,
            pypesto_result.optimize_result.list[start_index]["x"],
        )
    )
    x_dct.update(
        pypesto_result.problem.objective.calculator.necessary_par_dummy_values
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
    fill_in_parameters(
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
            "Warning: Some AMICI simulations failed. Cannot plot inner "
            "solutions.",
            stacklevel=2,
        )
        return None

    # Get simulation and sigma.
    sim = [rdata[AMICI_Y] for rdata in inner_rdatas]
    sigma = [rdata[AMICI_SIGMAY] for rdata in inner_rdatas]
    timepoints = [rdata[AMICI_T] for rdata in inner_rdatas]
    observable_ids = amici_model.getObservableIds()
    condition_ids = [edata.id for edata in edatas]
    petab_condition_ordering = list(petab_problem.condition_df.index)

    # Get the observable ordering from the measurement_df.
    measurement_df_observable_ordering = list(
        petab_problem.measurement_df[OBSERVABLE_ID].unique()
    )

    optimal_scaling_calculator = None
    for (
        calculator
    ) in pypesto_result.problem.objective.calculator.inner_calculators:
        if isinstance(calculator, OrdinalCalculator):
            optimal_scaling_calculator = calculator
            break

    # Get the inner solver and problem.
    inner_solver = optimal_scaling_calculator.inner_solver
    inner_problem = optimal_scaling_calculator.inner_problem

    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    return plot_categories_from_inner_result(
        inner_problem,
        inner_solver,
        inner_results,
        sim,
        timepoints,
        observable_ids,
        condition_ids,
        petab_condition_ordering,
        measurement_df_observable_ordering,
        axes,
        **kwargs,
    )


def plot_categories_from_inner_result(
    inner_problem: "pypesto.hierarchical.ordinal.problem.OrdinalProblem",
    inner_solver: "pypesto.hierarchical.ordinal.solver.OrdinalInnerSolver",
    results: list[dict],
    simulation: list[np.ndarray],
    timepoints: list[np.ndarray],
    observable_ids: list[str] = None,
    condition_ids: list[str] = None,
    petab_condition_ordering: list[str] = None,
    measurement_df_observable_ordering: list[str] = None,
    axes: Optional[plt.Axes] = None,
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
    axes:
        The optional axes to plot on.

    Returns
    -------
    fig:
        The figure.
    axes:
        The axes.
    """

    if len(results) != len(inner_problem.groups):
        raise ValueError(
            "Number of results must be equal to number of groups of the inner subproblem."
        )

    # Get the number of groups
    n_groups = len(inner_problem.groups)
    options = inner_solver.options

    use_given_axes = axes is not None

    # If there are no axes, make a figure with multiple plots
    if axes is None:
        axes = _get_default_axes(n_groups, **kwargs)

    # for each result and group, plot the inner solution
    for result, group in zip(results, inner_problem.groups):
        if observable_ids is not None and use_given_axes:
            observable_id = observable_ids[group - 1]
            meas_obs_idx = measurement_df_observable_ordering.index(
                observable_id
            )

            # Get the ax for the current observable.
            ax = axes["plot" + str(meas_obs_idx + 1)]
        else:
            ax = axes[list(inner_problem.groups.keys()).index(group)]

        # For each group get the inner parameters and simulation
        xs = inner_problem.get_cat_ub_parameters_for_group(group)

        interval_range, interval_gap = compute_interval_constraints(
            xs, simulation, options
        )
        observable_index = group - 1
        measurement_type = inner_problem.groups[group][MEASUREMENT_TYPE]
        # Get surrogate datapoints and category bounds
        (
            simulation_all,
            surrogate_all,
            timepoints_all,
            upper_bounds_all,
            lower_bounds_all,
        ) = _get_data_for_plotting(
            xs,
            result[SCIPY_X],
            simulation,
            timepoints,
            interval_range,
            interval_gap,
            options,
            measurement_type,
        )

        # Get the number of distinct timepoints in timepoints_all
        # where timepoints_all is a list of numpy arrays of timepoints
        n_distinct_timepoints = len(np.unique(np.concatenate(timepoints_all)))

        # If there is only one distinct timepoint, plot with respect to conditions
        if n_distinct_timepoints == 1 and not use_given_axes:
            _plot_observable_fit_across_conditions(
                ax,
                inner_problem,
                observable_index,
                group,
                condition_ids,
                simulation,
                simulation_all,
                surrogate_all,
                upper_bounds_all,
                lower_bounds_all,
                measurement_type,
                petab_condition_ordering,
                use_given_axes,
            )

        # Plotting across timepoints
        elif n_distinct_timepoints > 1:
            n_conditions = len(simulation_all)

            # If there is only one condition, we don't need
            # separate colors for the different conditions
            if n_conditions == 1:
                _plot_observable_fit_for_one_condition(
                    ax,
                    observable_index,
                    group,
                    inner_problem,
                    timepoints,
                    timepoints_all,
                    simulation,
                    simulation_all,
                    surrogate_all,
                    lower_bounds_all,
                    upper_bounds_all,
                    measurement_type,
                    use_given_axes,
                )

            # If there are multiple conditions, we need
            # separate colors for the different conditions
            elif n_conditions > 1:
                _plot_observable_fit_for_multiple_conditions(
                    ax,
                    observable_index,
                    group,
                    inner_problem,
                    timepoints,
                    timepoints_all,
                    simulation,
                    simulation_all,
                    surrogate_all,
                    lower_bounds_all,
                    upper_bounds_all,
                    measurement_type,
                    condition_ids,
                    use_given_axes,
                )

            ax.legend()

            if not use_given_axes:
                ax.set_title(f"Group {group}, {measurement_type} data")

            ax.set_xlabel("Timepoints")
            ax.set_ylabel("Simulation/Surrogate data")

    if not use_given_axes:
        for ax in axes[len(results) :]:
            ax.remove()

    return axes


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
            color="gray",
            alpha=0.5,
        )

    # Add to legend meaning of gray rectangles.
    ax.fill_between(
        [],
        [],
        [],
        color="gray",
        alpha=0.5,
        label="Categories",
    )


def _plot_category_rectangles(
    ax,
    timepoints,
    upper_bounds,
    lower_bounds,
    surrogate_data,
    measurement_type,
) -> None:
    """Plot the category rectangles."""
    interval_length = 0

    for i in range(len(timepoints)):
        if i + 1 == len(timepoints) or upper_bounds[i + 1] != upper_bounds[i]:
            if i + 1 == len(timepoints):
                if upper_bounds[i] == np.inf:
                    upper_bounds[i - interval_length : i + 1] = 1.1 * max(
                        surrogate_data
                    )
                    middle_index = int((i - interval_length + i) / 2)
                    middle_timepoint = timepoints[middle_index]
                    # Draw a vertical short grey arrow at the middle point of the interval
                    # at the upper_bounds[i] height
                    ax.annotate(
                        "",
                        xy=(middle_timepoint, upper_bounds[i]),
                        xytext=(
                            middle_timepoint,
                            upper_bounds[i] + 0.1 * max(surrogate_data),
                        ),
                        arrowprops={
                            "arrowstyle": "<-",
                            "color": "gray",
                            "linewidth": 2,
                        },
                    )
                    ax.text(
                        middle_timepoint,
                        upper_bounds[i] + 0.1 * max(surrogate_data),
                        "INF",
                        color="gray",
                        fontsize=12,
                    )
                    # Extend the ax to contain the text
                    ax.set_ylim(
                        bottom=ax.get_ylim()[0],
                        top=max(
                            ax.get_ylim()[1],
                            upper_bounds[i] + 0.1 * max(surrogate_data),
                        ),
                    )
                ax.fill_between(
                    timepoints[i - interval_length : i + 1],
                    upper_bounds[i - interval_length : i + 1],
                    lower_bounds[i - interval_length : i + 1],
                    color="gray",
                    alpha=0.5,
                )
            else:
                if upper_bounds[i] == np.inf:
                    upper_bounds[i - interval_length : i + 1] = 1.1 * max(
                        surrogate_data
                    )
                    middle_index = int((i - interval_length + i + 1) / 2)
                    middle_timepoint = timepoints[middle_index]
                    # Draw a vertical short grey arrow at the middle point of the interval
                    # at the upper_bounds[i] height
                    ax.annotate(
                        "",
                        xy=(middle_timepoint, upper_bounds[i]),
                        xytext=(
                            middle_timepoint,
                            upper_bounds[i] + 0.1 * max(surrogate_data),
                        ),
                        arrowprops={
                            "arrowstyle": "<-",
                            "color": "gray",
                            "linewidth": 2,
                        },
                    )
                    ax.text(
                        middle_timepoint,
                        upper_bounds[i] + 0.1 * max(surrogate_data),
                        "INF",
                        color="gray",
                        fontsize=12,
                    )
                    # Extend the ax to contain the text
                    ax.set_ylim(
                        bottom=ax.get_ylim()[0],
                        top=max(
                            ax.get_ylim()[1],
                            upper_bounds[i] + 0.1 * max(surrogate_data),
                        ),
                    )

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
                    color="gray",
                    alpha=0.5,
                )
            interval_length = 0
        else:
            interval_length += 1
    if measurement_type == ORDINAL:
        # Add to legend meaning of rectangles
        ax.fill_between(
            [],
            [],
            [],
            color="gray",
            alpha=0.5,
            label="Categories",
        )
    elif measurement_type == CENSORED:
        # Add to legend meaning of rectangles
        ax.fill_between(
            [],
            [],
            [],
            color="gray",
            alpha=0.5,
            label="Censoring areas",
        )


def _get_data_for_plotting(
    inner_parameters: list["OrdinalParameter"],
    optimal_scaling_bounds: list,
    sim: list[np.ndarray],
    timepoints: list[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: dict,
    measurement_type: str,
):
    """Return data in the form suited for plotting."""
    if options[REPARAMETERIZED] and measurement_type == ORDINAL:
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
            if measurement_type == ORDINAL:
                upper_bound, lower_bound = get_bounds_for_category(
                    inner_parameter,
                    optimal_scaling_bounds,
                    interval_gap,
                    options,
                )
            elif measurement_type == CENSORED:
                x_category = inner_parameter.category
                lower_bound = optimal_scaling_bounds[2 * x_category - 2]
                upper_bound = optimal_scaling_bounds[2 * x_category - 1]

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


def _get_default_axes(n_groups, **kwargs):
    """Return a list of axes with the default layout."""
    # Choose number of rows and columns to be used for the subplots
    n_rows = int(np.ceil(np.sqrt(n_groups)))
    n_cols = int(np.ceil(n_groups / n_rows))

    # Make as many subplots as there are groups
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, **kwargs)

    # Increase the spacing between the subplots
    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    # Flatten the axes array
    axes = axes.flatten()
    return axes


def _plot_observable_fit_across_conditions(
    ax,
    inner_problem,
    observable_index,
    group,
    condition_ids,
    simulation,
    simulation_all,
    surrogate_all,
    upper_bounds_all,
    lower_bounds_all,
    measurement_type,
    condition_ids_from_petab,
    use_given_axes,
):
    """Plot the observable fit across conditions.

    In case the observable has only one timepoint, the
    observable fit will be plotted against the conditions.
    """
    if measurement_type == CENSORED:
        # Get the condition indices which have censored data
        # and the corresponding condition ids with their ordering
        censored_condition_ids = [
            condition_ids[i]
            for i, cond_sim in enumerate(simulation_all)
            if len(cond_sim) > 0
        ]
        petab_censored_conditions = [
            condition_id
            for condition_id in condition_ids_from_petab
            if condition_id in censored_condition_ids
        ]
        petab_censored_conditions_ordering = [
            censored_condition_ids.index(condition_id)
            for condition_id in petab_censored_conditions
        ]
        # Get all other condition indices for quantitative data
        # and the corresponding condition ids with their ordering
        quantitative_condition_ids = [
            condition_id
            for condition_id in condition_ids
            if condition_id not in censored_condition_ids
        ]
        petab_quantitative_conditions = [
            condition_id
            for condition_id in condition_ids_from_petab
            if condition_id in quantitative_condition_ids
        ]
        petab_quantitative_condition_ordering = [
            quantitative_condition_ids.index(condition_id)
            for condition_id in petab_quantitative_conditions
        ]

    petab_condition_ordering = [
        condition_ids.index(condition_id)
        for condition_id in condition_ids_from_petab
    ]

    # Merge the simulation, surrogate, and bounds across conditions
    simulation_all = np.concatenate(simulation_all)
    surrogate_all = np.concatenate(surrogate_all)
    upper_bounds_all = np.concatenate(upper_bounds_all)
    lower_bounds_all = np.concatenate(lower_bounds_all)

    if measurement_type == CENSORED:
        # Change ordering of simulation, surrogate data and bounds to petab condition ordering
        simulation_all = simulation_all[petab_censored_conditions_ordering]
        surrogate_all = surrogate_all[petab_censored_conditions_ordering]
        upper_bounds_all = upper_bounds_all[petab_censored_conditions_ordering]
        lower_bounds_all = lower_bounds_all[petab_censored_conditions_ordering]

        whole_simulation = np.concatenate(
            [sim_i[:, observable_index] for sim_i in simulation]
        )[petab_condition_ordering]

        if not use_given_axes:
            ax.plot(
                condition_ids_from_petab,
                whole_simulation,
                linestyle="-",
                marker=".",
                color="b",
                label="Simulation",
            )
        ax.plot(
            petab_censored_conditions,
            surrogate_all,
            "rx",
            label="Surrogate data",
        )
        _plot_category_rectangles(
            ax,
            petab_censored_conditions,
            upper_bounds_all,
            lower_bounds_all,
            surrogate_all,
            measurement_type,
        )

        quantitative_data = inner_problem.groups[group][QUANTITATIVE_DATA]
        quantitative_data = quantitative_data[
            petab_quantitative_condition_ordering
        ]
        ax.plot(
            petab_quantitative_conditions,
            quantitative_data,
            "gs",
            label="Quantitative data",
        )

    elif measurement_type == ORDINAL:
        # Change ordering of simulation, surrogate data and bounds to petab condition ordering
        simulation_all = simulation_all[petab_condition_ordering]
        surrogate_all = surrogate_all[petab_condition_ordering]
        upper_bounds_all = upper_bounds_all[petab_condition_ordering]
        lower_bounds_all = lower_bounds_all[petab_condition_ordering]

        # Plot the categories and surrogate data across conditions
        if not use_given_axes:
            ax.plot(
                condition_ids_from_petab,
                simulation_all,
                linestyle="-",
                marker=".",
                color="b",
                label="Simulation",
            )
        ax.plot(
            condition_ids_from_petab,
            surrogate_all,
            "rx",
            label="Surrogate data",
        )

        _plot_category_rectangles(
            ax,
            condition_ids_from_petab,
            upper_bounds_all,
            lower_bounds_all,
            surrogate_all,
            measurement_type,
        )

    # Set the condition xticks on an angle
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    if not use_given_axes:
        ax.set_title(f"Group {group}, {measurement_type} data")

    ax.set_xlabel("Conditions")
    ax.set_ylabel("Simulation/Surrogate data")


def _plot_observable_fit_for_one_condition(
    ax,
    observable_index,
    group,
    inner_problem,
    timepoints,
    timepoints_all,
    simulation,
    simulation_all,
    surrogate_all,
    lower_bounds_all,
    upper_bounds_all,
    measurement_type,
    use_given_axes,
):
    """Plot the observable fit in case it has one condition."""
    if measurement_type == ORDINAL:
        if not use_given_axes:
            ax.plot(
                timepoints_all[0],
                simulation_all[0],
                linestyle="-",
                marker=".",
                color="b",
                label="Simulation",
            )
    elif measurement_type == CENSORED:
        quantitative_data = inner_problem.groups[group][QUANTITATIVE_DATA]
        quantitative_ixs = inner_problem.groups[group][QUANTITATIVE_IXS]
        quantitative_timepoints = timepoints[0][
            quantitative_ixs[0].T[observable_index]
        ]

        if not use_given_axes:
            ax.plot(
                timepoints[0],
                simulation[0][:, observable_index],
                linestyle="-",
                marker=".",
                color="b",
                label="Simulation",
            )
        ax.plot(
            quantitative_timepoints,
            quantitative_data,
            "gs",
            label="Quantitative data",
        )

    ax.plot(
        timepoints_all[0],
        surrogate_all[0],
        "rx",
        label="Surrogate data",
    )

    # Plot the categorie rectangles
    _plot_category_rectangles(
        ax,
        timepoints_all[0],
        upper_bounds_all[0],
        lower_bounds_all[0],
        surrogate_all[0],
        measurement_type,
    )


def _plot_observable_fit_for_multiple_conditions(
    ax,
    observable_index,
    group,
    inner_problem,
    timepoints,
    timepoints_all,
    simulation,
    simulation_all,
    surrogate_all,
    lower_bounds_all,
    upper_bounds_all,
    measurement_type,
    condition_ids,
    use_given_axes,
):
    """Plot the observable fit in case it has multiple conditions."""
    # Get the colors from the plotted simulations
    if use_given_axes:
        colors = []
        for line in ax.lines:
            if "simulation" in line.get_label():
                colors.append(line.get_color())
    # Get as many colors as there are conditions
    else:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(simulation_all)))

    if measurement_type == CENSORED:
        quantitative_data_flattened = inner_problem.groups[group][
            QUANTITATIVE_DATA
        ]
        quantitative_ixs = inner_problem.groups[group][QUANTITATIVE_IXS]
        quantitative_timepoints = [
            timepoints[cond_i][quantitative_ixs[cond_i].T[observable_index]]
            for cond_i in range(len(timepoints))
        ]
        quantitative_data = []
        index_offset = 0
        # Separate quantitative data across conditions to be as timepoints
        for cond_i in range(len(timepoints)):
            quantitative_data.append(
                quantitative_data_flattened[
                    index_offset : index_offset
                    + len(quantitative_timepoints[cond_i])
                ]
            )

    # Plot the categories and surrogate data for all conditions.
    for condition_index, condition_id, color in zip(
        range(len(simulation_all)), condition_ids, colors
    ):
        # Plot the categories and surrogate data for the current condition
        if measurement_type == ORDINAL:
            if not use_given_axes:
                ax.plot(
                    timepoints_all[condition_index],
                    simulation_all[condition_index],
                    linestyle="-",
                    marker=".",
                    color=color,
                    label=condition_id,
                )
        elif measurement_type == CENSORED:
            if not use_given_axes:
                ax.plot(
                    timepoints[condition_index],
                    simulation[condition_index][:, observable_index],
                    linestyle="-",
                    marker=".",
                    color=color,
                    label=condition_id,
                )
            ax.plot(
                quantitative_timepoints[condition_index],
                quantitative_data[condition_index],
                marker="s",
                color=color,
            )

        ax.plot(
            timepoints_all[condition_index],
            surrogate_all[condition_index],
            "x",
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
                category_timepoints_dict[(upper_bound, lower_bound)] = [
                    timepoint
                ]
            else:
                category_timepoints_dict[(upper_bound, lower_bound)].append(
                    timepoint
                )

    # Plot the category rectangles
    _plot_category_rectangles_across_conditions(
        ax,
        category_timepoints_dict,
        unique_timepoints,
    )

    # Add to legend meaning of x, and -o- markers.
    ax.plot(
        [],
        [],
        "x",
        color="black",
        label="Surrogate data",
    )
    if not use_given_axes:
        ax.plot(
            [],
            [],
            linestyle="-",
            marker=".",
            color="black",
            label="Simulation",
        )
    if measurement_type == CENSORED:
        ax.plot(
            [],
            [],
            marker="s",
            color="black",
            label="Quantitative data",
        )
