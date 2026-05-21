from __future__ import annotations

import warnings
from os.path import commonprefix
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pypesto

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import amici.sim.sundials as asd
    from amici.sim.sundials.petab.v1 import fill_in_parameters
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
from .misc import get_axes_array, hide_unused_axes, make_grid_shape
from ._style import resolve_style


def plot_categories_from_pypesto_result(
    pypesto_result: Result,
    start_index=0,
    axes: plt.Axes | np.ndarray | dict[str, plt.Axes] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    style_kwargs: dict | None = None,
) -> np.ndarray | dict[str, plt.Axes] | None:
    """Plot the inner solutions from a pypesto result.

    Parameters
    ----------
    pypesto_result:
        The pypesto result.
    start_index:
        The index of the pypesto_result.optimize_result.list to plot.
    axes:
        Optional axes to plot on. Pass a normal Axes array for standalone
        reuse, or the dict returned by PEtab's model-fit plot for overlaying
        ordinal annotations on an existing model-fit figure.
    size:
        Figure size ``(width, height)`` in inches; only used when ``axes`` is
        ``None``. Defaults to a grid-scaled size.
    title:
        Figure title.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``line_color`` — simulation line color.
        - ``mle_color`` — surrogate-data marker color.
        - ``data_color`` — quantitative-data marker color.
        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — category/censoring rectangle styling.
        - ``trace_linewidth``, ``line_marker_size``, ``marker_linewidth`` —
          line and marker geometry.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        The axes used for plotting, or ``None`` if the simulations failed.
    """

    style = resolve_style(style_kwargs)

    # Get the parameters from the pypesto result for the start_index.
    x_dct = dict(
        zip(
            pypesto_result.problem.objective.x_ids,
            pypesto_result.optimize_result.list[start_index]["x"],
            strict=True,
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
    inner_rdatas = asd.run_simulations(
        amici_model,
        amici_solver,
        edatas,
        num_threads=min(n_threads, len(edatas)),
    )

    # If any amici simulation failed, raise warning and return None.
    if any(rdata.status != asd.AMICI_SUCCESS for rdata in inner_rdatas):
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
    observable_ids = amici_model.get_observable_ids()
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
        size=size,
        title=title,
        style_kwargs=style,
    )


def plot_categories_from_inner_result(
    inner_problem: "pypesto.hierarchical.ordinal.problem.OrdinalProblem",
    inner_solver: "pypesto.hierarchical.ordinal.solver.OrdinalInnerSolver",
    results: list[dict],
    simulation: list[np.ndarray],
    timepoints: list[np.ndarray],
    observable_ids: list[str] | None = None,
    condition_ids: list[str] | None = None,
    petab_condition_ordering: list[str] | None = None,
    measurement_df_observable_ordering: list[str] | None = None,
    axes: plt.Axes | np.ndarray | dict[str, plt.Axes] | None = None,
    size: tuple[float, float] | None = None,
    title: str | None = None,
    style_kwargs: dict | None = None,
) -> np.ndarray | dict[str, plt.Axes]:
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
    axes:
        Optional axes to plot on. Pass a normal Axes array for standalone
        reuse, or the dict returned by PEtab's model-fit plot for overlaying
        ordinal annotations on an existing model-fit figure.
    size:
        Figure size ``(width, height)`` in inches; only used when ``axes`` is
        ``None``. Defaults to a grid-scaled size.
    title:
        Figure title.
    style_kwargs:
        Style overrides. Keys used by this function:

        - ``line_color`` — simulation line color.
        - ``mle_color`` — surrogate-data marker color.
        - ``data_color`` — quantitative-data marker color.
        - ``rectangle_color``, ``rectangle_alpha``, ``rectangle_edgecolor``,
          ``rectangle_linewidth`` — category/censoring rectangle styling.
        - ``trace_linewidth``, ``line_marker_size``, ``marker_linewidth`` —
          line and marker geometry.

        All valid keys and their defaults are listed in
        :data:`pypesto.visualize._style._DEFAULTS`.

    Returns
    -------
    axes:
        The axes.
    """

    style = resolve_style(style_kwargs)

    if len(results) != len(inner_problem.groups):
        raise ValueError(
            "Number of results must be equal to number of groups of the inner subproblem."
        )

    # Get the number of groups
    n_groups = len(inner_problem.groups)
    options = inner_solver.options

    overlay_model_fit_axes = isinstance(axes, dict)
    created_axes = axes is None

    if axes is None:
        n_rows, n_cols = make_grid_shape(n_groups)
        axes = get_axes_array(nrows=n_rows, ncols=n_cols, size=size)
        axes = hide_unused_axes(axes=axes, n_used=n_groups, clear=True)
    elif not overlay_model_fit_axes:
        axes = np.asarray(axes, dtype=object)
        if axes.ndim == 0:
            axes = axes.reshape((1, 1))
        elif axes.ndim == 1:
            axes = axes.reshape((1, axes.size))
        if axes.size < n_groups:
            raise ValueError(
                f"Expected at least {n_groups} axes, got {axes.size}."
            )

    # for each result and group, plot the inner solution
    for idx, (result, group) in enumerate(
        zip(results, inner_problem.groups, strict=True)
    ):
        if overlay_model_fit_axes:
            if (
                observable_ids is None
                or measurement_df_observable_ordering is None
            ):
                raise ValueError(
                    "Observable metadata is required when plotting ordinal "
                    "categories into PEtab model-fit axes."
                )
            observable_id = observable_ids[group - 1]
            meas_obs_idx = measurement_df_observable_ordering.index(
                observable_id
            )

            # Get the ax for the current observable.
            ax = axes["plot" + str(meas_obs_idx + 1)]
        else:
            ax = axes.flat[list(inner_problem.groups.keys()).index(group)]
        show_legend = idx == 0

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
        if n_distinct_timepoints == 1 and not overlay_model_fit_axes:
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
                overlay_model_fit_axes,
                style,
                show_legend,
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
                    overlay_model_fit_axes,
                    style,
                    show_legend,
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
                    overlay_model_fit_axes,
                    style,
                    show_legend,
                )

            if not overlay_model_fit_axes:
                ax.set_title(f"Group {group}")

            ax.set_xlabel("Timepoints")
            ax.set_ylabel("Value")

    if not overlay_model_fit_axes:
        n_rows, n_cols = make_grid_shape(n_groups)
        for idx, ax in enumerate(axes.flat):
            if idx >= n_groups:
                if created_axes:
                    ax.set_visible(False)
                continue
            if idx % n_cols != 0:
                ax.set_ylabel("")
            if n_rows > 1 and idx // n_cols < n_rows - 1:
                ax.set_xlabel("")

    if title is not None:
        if isinstance(axes, dict):
            first_ax = next(iter(axes.values()))
        else:
            first_ax = np.asarray(axes, dtype=object).flat[0]
        first_ax.figure.suptitle(title)

    return axes


def _plot_category_rectangles_across_conditions(
    ax, category_timepoints_dict, unique_timepoints, style
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
            facecolor=style["rectangle_color"],
            edgecolor=style["rectangle_edgecolor"],
            linewidth=style["rectangle_linewidth"],
            alpha=style["rectangle_alpha"],
        )


def _plot_category_rectangles(
    ax,
    timepoints,
    upper_bounds,
    lower_bounds,
    surrogate_data,
    measurement_type,
    style,
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
                            "color": style["rectangle_color"],
                            "linewidth": style["trace_linewidth"],
                        },
                    )
                    ax.text(
                        middle_timepoint,
                        upper_bounds[i] + 0.1 * max(surrogate_data),
                        "INF",
                        color=style["rectangle_color"],
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
                    facecolor=style["rectangle_color"],
                    edgecolor=style["rectangle_edgecolor"],
                    linewidth=style["rectangle_linewidth"],
                    alpha=style["rectangle_alpha"],
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
                            "color": style["rectangle_color"],
                            "linewidth": style["trace_linewidth"],
                        },
                    )
                    ax.text(
                        middle_timepoint,
                        upper_bounds[i] + 0.1 * max(surrogate_data),
                        "INF",
                        color=style["rectangle_color"],
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
                    facecolor=style["rectangle_color"],
                    edgecolor=style["rectangle_edgecolor"],
                    linewidth=style["rectangle_linewidth"],
                    alpha=style["rectangle_alpha"],
                )
            interval_length = 0
        else:
            interval_length += 1


def _format_condition_axis(condition_ids) -> tuple[list[str], str]:
    """Return compact tick labels and an x-axis label for condition IDs."""
    labels = [str(condition_id) for condition_id in condition_ids]
    if len(labels) < 2:
        return labels, "Conditions"

    separators = ("_", "-", " ", ":")
    prefix = commonprefix(labels)
    prefix_cut = max(prefix.rfind(separator) for separator in separators)
    prefix_part = labels[0][:prefix_cut] if prefix_cut > 0 else ""
    prefix_len = prefix_cut + 1 if prefix_cut > 0 else 0

    reversed_labels = [label[::-1] for label in labels]
    suffix_reversed = commonprefix(reversed_labels)
    suffix = suffix_reversed[::-1]
    suffix_cut = min(
        (suffix.find(separator) for separator in separators if separator in suffix),
        default=-1,
    )
    suffix_part = suffix[suffix_cut + 1 :] if suffix_cut >= 0 else ""
    suffix_len = len(suffix) - suffix_cut if suffix_cut >= 0 else 0

    if prefix_len == 0 and suffix_len == 0:
        return labels, "Conditions"

    tick_labels = [
        label[prefix_len : len(label) - suffix_len if suffix_len else len(label)]
        for label in labels
    ]
    if any(not label for label in tick_labels):
        return labels, "Conditions"
    if len(set(tick_labels)) != len(tick_labels):
        return labels, "Conditions"
    if sum(map(len, tick_labels)) > 0.75 * sum(map(len, labels)):
        return labels, "Conditions"

    axis_parts = [part for part in (prefix_part, suffix_part) if part]
    axis_label = " / ".join(axis_parts) if axis_parts else "Conditions"
    return tick_labels, axis_label


def _condition_tick_rotation(tick_labels: list[str]) -> int:
    """Return a readable rotation for formatted condition tick labels."""
    try:
        for label in tick_labels:
            float(label)
        return 0
    except ValueError:
        return 25


def _add_ordinal_legend(
    ax,
    *,
    style: dict,
    measurement_type: str,
    show_simulation: bool,
    show_quantitative: bool,
) -> None:
    """Draw one compact semantic legend for ordinal/censored panels."""
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    semantic_labels = {"Simulation", "Surrogate data", "Quantitative data"}
    handles = [
        handle
        for handle, label in zip(existing_handles, existing_labels, strict=True)
        if label and label not in semantic_labels and not label.startswith("_")
    ]
    labels = [
        label
        for label in existing_labels
        if label and label not in semantic_labels and not label.startswith("_")
    ]

    if show_simulation and "Simulation" not in labels:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["line_color"],
                marker="o",
                linewidth=style["trace_linewidth"],
                markersize=style["line_marker_size"],
            )
        )
        labels.append("Simulation")

    handles.append(
        Line2D(
            [0],
            [0],
            color=style["mle_color"],
            marker="D",
            linestyle="none",
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
        )
    )
    labels.append("Surrogate data")

    if show_quantitative:
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["data_color"],
                marker="s",
                linestyle="none",
                markersize=style["line_marker_size"],
                markeredgewidth=style["marker_linewidth"],
            )
        )
        labels.append("Quantitative data")

    handles.append(
        Patch(
            facecolor=style["rectangle_color"],
            edgecolor=style["rectangle_edgecolor"],
            linewidth=style["rectangle_linewidth"],
            alpha=style["rectangle_alpha"],
        )
    )
    labels.append("Categories" if measurement_type == ORDINAL else "Censoring areas")

    ax.legend(handles=handles, labels=labels)


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
    style,
    show_legend,
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
    condition_positions = {
        condition_id: i for i, condition_id in enumerate(condition_ids_from_petab)
    }
    x_all = np.arange(len(condition_ids_from_petab))

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
                x_all,
                whole_simulation,
                linestyle="-",
                marker="o",
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
                markersize=style["line_marker_size"],
                label="Simulation",
            )
        ax.plot(
            [condition_positions[condition] for condition in petab_censored_conditions],
            surrogate_all,
            linestyle="none",
            marker="D",
            color=style["mle_color"],
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
            label="Surrogate data",
        )
        _plot_category_rectangles(
            ax,
            np.array(
                [condition_positions[condition] for condition in petab_censored_conditions]
            ),
            upper_bounds_all,
            lower_bounds_all,
            surrogate_all,
            measurement_type,
            style,
        )

        quantitative_data = inner_problem.groups[group][QUANTITATIVE_DATA]
        quantitative_data = quantitative_data[
            petab_quantitative_condition_ordering
        ]
        ax.plot(
            [
                condition_positions[condition]
                for condition in petab_quantitative_conditions
            ],
            quantitative_data,
            linestyle="none",
            marker="s",
            color=style["data_color"],
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
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
                x_all,
                simulation_all,
                linestyle="-",
                marker="o",
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
                markersize=style["line_marker_size"],
                label="Simulation",
            )
        ax.plot(
            x_all,
            surrogate_all,
            linestyle="none",
            marker="D",
            color=style["mle_color"],
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
            label="Surrogate data",
        )

        _plot_category_rectangles(
            ax,
            x_all,
            upper_bounds_all,
            lower_bounds_all,
            surrogate_all,
            measurement_type,
            style,
        )

    tick_labels, x_label = _format_condition_axis(condition_ids_from_petab)
    ax.set_xticks(x_all)
    ax.set_xticklabels(tick_labels)
    rotation = _condition_tick_rotation(tick_labels)
    ax.tick_params(axis="x", rotation=rotation)
    for tick_label in ax.get_xticklabels():
        tick_label.set_ha("center" if rotation == 0 else "right")
    if show_legend:
        _add_ordinal_legend(
            ax,
            style=style,
            measurement_type=measurement_type,
            show_simulation=not use_given_axes,
            show_quantitative=(measurement_type == CENSORED),
        )
    if not use_given_axes:
        ax.set_title(f"Group {group}")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Value")


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
    style,
    show_legend,
):
    """Plot the observable fit in case it has one condition."""
    if measurement_type == ORDINAL:
        if not use_given_axes:
            ax.plot(
                timepoints_all[0],
                simulation_all[0],
                linestyle="-",
                marker="o",
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
                markersize=style["line_marker_size"],
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
                marker="o",
                color=style["line_color"],
                linewidth=style["trace_linewidth"],
                markersize=style["line_marker_size"],
                label="Simulation",
            )
        ax.plot(
            quantitative_timepoints,
            quantitative_data,
            linestyle="none",
            marker="s",
            color=style["data_color"],
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
            label="Quantitative data",
        )

    ax.plot(
        timepoints_all[0],
        surrogate_all[0],
        linestyle="none",
        marker="D",
        color=style["mle_color"],
        markersize=style["line_marker_size"],
        markeredgewidth=style["marker_linewidth"],
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
        style,
    )
    if show_legend:
        _add_ordinal_legend(
            ax,
            style=style,
            measurement_type=measurement_type,
            show_simulation=not use_given_axes,
            show_quantitative=(measurement_type == CENSORED),
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
    style,
    show_legend,
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
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [
            color_cycle[i % len(color_cycle)]
            for i in range(len(simulation_all))
        ]

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
        range(len(simulation_all)), condition_ids, colors, strict=True
    ):
        # Plot the categories and surrogate data for the current condition
        if measurement_type == ORDINAL:
            if not use_given_axes:
                ax.plot(
                    timepoints_all[condition_index],
                    simulation_all[condition_index],
                    linestyle="-",
                    marker="o",
                    color=color,
                    linewidth=style["trace_linewidth"],
                    markersize=style["line_marker_size"],
                    label=condition_id,
                )
        elif measurement_type == CENSORED:
            if not use_given_axes:
                ax.plot(
                    timepoints[condition_index],
                    simulation[condition_index][:, observable_index],
                    linestyle="-",
                    marker="o",
                    color=color,
                    linewidth=style["trace_linewidth"],
                    markersize=style["line_marker_size"],
                    label=condition_id,
                )
            ax.plot(
                quantitative_timepoints[condition_index],
                quantitative_data[condition_index],
                marker="s",
                linestyle="none",
                color=style["data_color"],
                markersize=style["line_marker_size"],
                markeredgewidth=style["marker_linewidth"],
            )

        ax.plot(
            timepoints_all[condition_index],
            surrogate_all[condition_index],
            linestyle="none",
            marker="D",
            color=style["mle_color"],
            markersize=style["line_marker_size"],
            markeredgewidth=style["marker_linewidth"],
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
            strict=True,
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
        style,
    )

    if show_legend:
        _add_ordinal_legend(
            ax,
            style=style,
            measurement_type=measurement_type,
            show_simulation=not use_given_axes,
            show_quantitative=(measurement_type == CENSORED),
        )
