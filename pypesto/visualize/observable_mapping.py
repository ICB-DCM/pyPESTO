import warnings
from collections.abc import Sequence

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

import pypesto

from ..C import (
    AMICI_SIGMAY,
    AMICI_Y,
    CURRENT_SIMULATION,
    DATAPOINTS,
    EXPDATA_MASK,
    INNER_PARAMETERS,
    REGULARIZE_SPLINE,
    SCIPY_X,
    SPLINE_KNOTS,
    InnerParameterType,
)
from ..problem import HierarchicalProblem, Problem
from ..result import Result
from .misc import get_axes_array, hide_unused_axes, make_grid_shape

try:
    import amici.sim.sundials as asd
    from amici.sim.sundials.petab.v1 import fill_in_parameters

    from ..hierarchical import InnerCalculatorCollector
    from ..hierarchical.base_problem import scale_back_value_dict
    from ..hierarchical.relative.calculator import RelativeAmiciCalculator
    from ..hierarchical.relative.problem import RelativeInnerProblem
    from ..hierarchical.semiquantitative.calculator import SemiquantCalculator
    from ..hierarchical.semiquantitative.solver import (
        SemiquantInnerSolver,
        _calculate_regularization_for_group,
        extract_expdata_using_mask,
        get_spline_mapped_simulations,
    )
except ImportError:
    pass


def _prepare_observable_mapping_axes(
    axes: matplotlib.axes.Axes | np.ndarray | None,
    n_panels: int,
    **kwargs,
) -> np.ndarray:
    """Return a cleared axes grid for observable-mapping plots."""
    n_rows, n_cols = make_grid_shape(n_panels)

    if axes is None:
        kwargs.setdefault("layout", "constrained")
        _, axes = plt.subplots(n_rows, n_cols, squeeze=False, **kwargs)
    else:
        axes = get_axes_array(axes=axes, nrows=n_rows, ncols=n_cols)

    return hide_unused_axes(axes=axes, n_used=n_panels, clear=True)


def _plot_observable_mapping_measurements(
    ax: matplotlib.axes.Axes,
    simulation: np.ndarray,
    measurements: np.ndarray,
) -> None:
    """Plot measurement points with the shared visualization styling."""
    ax.scatter(
        simulation,
        measurements,
        color="C0",
        s=40,
        alpha=0.9,
        linewidths=0.6,
        edgecolors="white",
        label="Measurements",
        zorder=3,
    )


def _finalize_observable_mapping_axes(
    ax: matplotlib.axes.Axes,
    title: str,
) -> None:
    """Apply consistent styling to observable-mapping panels."""
    ax.set_title(title)
    ax.set_xlabel("Model output")
    ax.set_ylabel("Measurements")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05, y=0.08)
    ax.legend(frameon=False, loc="best")


def visualize_estimated_observable_mapping(
    pypesto_result: Result,
    pypesto_problem: HierarchicalProblem,
    start_index: int = 0,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    **kwargs,
) -> np.ndarray | None:
    """Visualize the estimated observable mapping for relative and semi-quantitative observables.

    Visualizes the estimated linear mapping for relative observables and the non-linear
    spline approximation for semi-quantitative observables.

    Parameters
    ----------
    pypesto_result:
        The pyPESTO result object from optimization.
    pypesto_problem:
        The pyPESTO problem. It should contain the objective object that was used for estimation.
    start_index:
        The observable mapping from this start's optimized vector will be plotted.
    axes:
        Optional axes grid to draw into.
    kwargs:
        Additional arguments to passed to ``matplotlib.pyplot.subplots``
        (e.g. `figsize= ...`).

    Returns
    -------
    axes:
        A 2-D NumPy array of matplotlib Axes, or ``None`` if the required
        simulation fails.
    """

    # Check if the pyPESTO problem is hierarchical.
    if not isinstance(pypesto_problem, HierarchicalProblem):
        raise ValueError(
            "Only hierarchical problems contain estimated observable mappings. Please provide a hierarchical problem."
        )

    # Check if the pyPESTO problem contains an objective.
    if pypesto_problem.objective is None:
        raise ValueError(
            "The problem must contain the corresponding objective that was used for estimation."
        )

    # Check the calculator is the InnerCalculatorCollector.
    if not isinstance(
        pypesto_problem.objective.calculator, InnerCalculatorCollector
    ):
        raise ValueError(
            "The calculator must be an instance of the InnerCalculatorCollector."
        )

    amici_model = pypesto_problem.objective.amici_model

    # Get the number of relative and semi-quantitative observables.
    relative_obs_ids = pypesto_problem.relative_observable_ids or []
    semiquant_obs_ids = pypesto_problem.semiquant_observable_ids or []
    n_relative_observables = len(relative_obs_ids)
    n_semiquant_observables = len(semiquant_obs_ids)

    # Check if there are any relative or semi-quantitative observables.
    if n_relative_observables == 0 and n_semiquant_observables == 0:
        raise ValueError(
            "The problem does not contain any relative or semi-quantitative observables."
        )

    # Get observable indices for both relative and semi-quantitative observables.
    rel_and_semiquant_obs_indices = [
        amici_model.get_observable_ids().index(observable_id)
        for observable_id in relative_obs_ids + semiquant_obs_ids
    ]
    rel_and_semiquant_obs_indices.sort()

    n_axes = n_relative_observables + n_semiquant_observables
    axes = _prepare_observable_mapping_axes(
        axes=axes,
        n_panels=n_axes,
        **kwargs,
    )

    # Plot the estimated observable mapping for relative observables.
    if n_relative_observables > 0:
        axes = plot_linear_observable_mappings_from_pypesto_result(
            pypesto_result=pypesto_result,
            pypesto_problem=pypesto_problem,
            start_index=start_index,
            axes=axes,
            rel_and_semiquant_obs_indices=rel_and_semiquant_obs_indices,
        )

    # Plot the estimated spline approximations for semi-quantitative observables.
    if n_semiquant_observables > 0:
        spline_axes = plot_splines_from_pypesto_result(
            pypesto_result=pypesto_result,
            start_index=start_index,
            axes=axes,
            rel_and_semiquant_obs_indices=rel_and_semiquant_obs_indices,
        )
        if spline_axes is None:
            return None
        axes = spline_axes

    return axes


def plot_linear_observable_mappings_from_pypesto_result(
    pypesto_result: Result,
    pypesto_problem: HierarchicalProblem,
    start_index=0,
    axes: np.ndarray | None = None,
    rel_and_semiquant_obs_indices: list[int] | None = None,
    **kwargs,
) -> np.ndarray:
    """Plot the linear observable mappings from a pyPESTO result.

    Parameters
    ----------
    pypesto_result:
        The pyPESTO result object from optimization.
    pypesto_problem:
        The pyPESTO problem. It should contain the objective object that was used for estimation.
    start_index:
        The observable mapping from this start's optimized vector will be plotted.
    axes:
        Optional 2-D NumPy array of Axes to draw into. Required when
        ``rel_and_semiquant_obs_indices`` is set.
    rel_and_semiquant_obs_indices:
        Sorted indices of the relative and semi-quantitative observables in
        the AMICI model. Each observable is plotted on the subplot at the
        corresponding position in ``axes.flat``.
    **kwargs:
        Additional arguments to pass to the ``matplotlib.pyplot.subplots`` function.

    Returns
    -------
    axes:
        A 2-D NumPy array of matplotlib Axes.
    """
    # Check the calculator is the InnerCalculatorCollector.
    if not isinstance(
        pypesto_problem.objective.calculator, InnerCalculatorCollector
    ):
        raise ValueError(
            "The calculator must be an instance of the InnerCalculatorCollector."
        )

    # Get the needed objects from the pypesto problem.
    edatas = pypesto_problem.objective.edatas
    parameter_mapping = pypesto_problem.objective.parameter_mapping
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    n_threads = pypesto_problem.objective.n_threads

    # Get the relative calculator.
    relative_calculator = [
        inner_calculator
        for inner_calculator in pypesto_problem.objective.calculator.inner_calculators
        if isinstance(inner_calculator, RelativeAmiciCalculator)
    ][0]

    # Get the inner problem and solver from the relative calculator.
    inner_problem: RelativeInnerProblem = relative_calculator.inner_problem

    # Get the relative observable ids and indices.
    relative_observable_ids = pypesto_problem.relative_observable_ids or []
    relative_observable_indices = [
        amici_model.get_observable_ids().index(observable_id)
        for observable_id in relative_observable_ids
    ]

    # Get the number of relative observables.
    n_relative_observables = len(relative_observable_ids)
    if n_relative_observables == 0:
        raise ValueError(
            "The problem does not contain any relative observables."
        )

    if rel_and_semiquant_obs_indices is None:
        axes = _prepare_observable_mapping_axes(
            axes=axes,
            n_panels=n_relative_observables,
            **kwargs,
        )
    else:
        if axes is None:
            raise ValueError(
                "Pass `axes` when `rel_and_semiquant_obs_indices` is set."
            )
        if not isinstance(axes, np.ndarray) or axes.ndim != 2:
            raise ValueError("`axes` must be a 2-D NumPy array.")
        if axes.size < len(rel_and_semiquant_obs_indices):
            raise ValueError(
                "The number of axes must be at least equal to the number "
                "of relative and semi-quantitative observables."
            )

    flat_axes = axes.flat

    #################################################################
    # Simulate the model with the parameters from the pypesto result.
    #################################################################

    # Get the parameters from the pypesto result for the start_index.
    x_dct = dict(
        zip(
            pypesto_problem.objective.x_ids,
            pypesto_result.optimize_result.list[start_index]["x"],
            strict=True,
        )
    )

    x_dct.update(
        pypesto_result.problem.objective.calculator.necessary_par_dummy_values
    )

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
        raise ValueError(
            "Warning: Some AMICI simulations failed. Cannot plot inner "
            "solutions."
        )

    # Get the simulation.
    sim = [rdata[AMICI_Y] for rdata in inner_rdatas]

    # Get the inner parameters from the pypesto result.
    inner_parameter_values = dict(
        zip(
            pypesto_problem.inner_x_names,
            pypesto_result.optimize_result.list[start_index][INNER_PARAMETERS],
            strict=True,
        )
    )

    # Remove inner parameters not belonging to the relative inner problem.
    inner_parameter_values = {
        key: value
        for key, value in inner_parameter_values.items()
        if key in inner_problem.get_x_ids()
    }

    # Scale the inner parameters back to linear scale.
    inner_parameter_values = scale_back_value_dict(
        inner_parameter_values, inner_problem
    )

    ######################################
    # Plot the linear observable mappings.
    ######################################

    # plot the linear observable mapping for each relative_observable_id
    for observable_index, observable_id in zip(
        relative_observable_indices, relative_observable_ids, strict=True
    ):
        # Get the ax for the current observable.
        if rel_and_semiquant_obs_indices is not None:
            ax_index = rel_and_semiquant_obs_indices.index(observable_index)
        else:
            ax_index = relative_observable_indices.index(observable_index)

        ax = flat_axes[ax_index]

        # Get the inner parameters for the current observable.
        inner_parameters = inner_problem.get_xs_for_obs_idx(observable_index)

        scaling_factor = None
        offset = None

        for inner_par in inner_parameters:
            if inner_par.inner_parameter_type == InnerParameterType.SCALING:
                scaling_factor = inner_par
            elif inner_par.inner_parameter_type == InnerParameterType.OFFSET:
                offset = inner_par

        scaling_factor_value = (
            inner_parameter_values[scaling_factor.inner_parameter_id]
            if scaling_factor is not None
            else 1
        )
        offset_value = (
            inner_parameter_values[offset.inner_parameter_id]
            if offset is not None
            else 0
        )

        # Get the data mask for the current observable.
        observable_data_mask = scaling_factor.ixs or offset.ixs

        # Get the measurements for the current observable.
        measurements = extract_expdata_using_mask(
            expdata=inner_problem.data, mask=observable_data_mask
        )

        # Get the simulation for the current observable.
        simulation = extract_expdata_using_mask(
            expdata=sim, mask=observable_data_mask
        )

        _plot_observable_mapping_measurements(ax, simulation, measurements)

        # Plot the linear mapping.
        sorted_simulation = np.sort(simulation)
        ax.plot(
            sorted_simulation,
            scaling_factor_value * sorted_simulation + offset_value,
            linestyle="-",
            color="C1",
            linewidth=1.8,
            label="Linear mapping",
            zorder=2,
        )

        _finalize_observable_mapping_axes(
            ax,
            title=f"Observable {observable_id}",
        )

    return axes


def plot_splines_from_pypesto_result(
    pypesto_result: Result,
    start_index=0,
    axes: matplotlib.axes.Axes | np.ndarray | None = None,
    rel_and_semiquant_obs_indices: list[int] | None = None,
    **kwargs,
) -> np.ndarray | None:
    """Plot the estimated spline approximations from a pypesto result.

    Parameters
    ----------
    pypesto_result:
        The pypesto result.
    start_index:
        The observable mapping from this start's optimized vector will be plotted.
    axes:
        Optional axes grid to draw into.
    rel_and_semiquant_obs_indices:
        The indices of the relative and semi-quantitative observables in the
        amici model. Important if both relative and semi-quantitative observables
        will be plotted on the same axes.
    kwargs:
        Additional arguments to pass to the plotting function.

    Returns
    -------
    axes:
        A 2-D NumPy array of matplotlib Axes, or ``None`` if the required
        simulation fails.
    """
    # Check that the problem contains an objective.
    if pypesto_result.problem.objective is None:
        raise ValueError(
            "The problem must contain the corresponding objective that was used for estimation."
        )

    # Check the calculator is the InnerCalculatorCollector.
    if not isinstance(
        pypesto_result.problem.objective.calculator, InnerCalculatorCollector
    ):
        raise ValueError(
            "The calculator must be an instance of the InnerCalculatorCollector."
        )

    # Check the result for start index contains the spline knots.
    if SPLINE_KNOTS not in pypesto_result.optimize_result.list[start_index]:
        raise ValueError(
            f"The result with index {start_index} does not contain the spline knots."
        )

    # Get the spline knot values from the pypesto result
    spline_knot_values = [
        obs_spline_knots[1]
        for obs_spline_knots in pypesto_result.optimize_result.list[
            start_index
        ][SPLINE_KNOTS]
    ]

    # Get inner parameters per observable as differences of spline knot values
    inner_parameters = [
        np.concatenate([[obs_knot_values[0]], np.diff(obs_knot_values)])
        for obs_knot_values in spline_knot_values
    ]

    inner_results = [
        {SCIPY_X: obs_inner_parameter}
        for obs_inner_parameter in inner_parameters
    ]

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
    n_threads = pypesto_result.problem.objective.n_threads
    observable_ids = amici_model.get_observable_ids()

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

    # Get simulation.
    sim = [rdata[AMICI_Y] for rdata in inner_rdatas]

    spline_calculator = None
    for (
        calculator
    ) in pypesto_result.problem.objective.calculator.inner_calculators:
        if isinstance(calculator, SemiquantCalculator):
            spline_calculator = calculator
            break

    if spline_calculator is None:
        raise ValueError(
            "No SemiquantCalculator found in the inner_calculators of the objective. "
            "Cannot plot splines."
        )

    # Get the inner solver and problem.
    inner_solver = spline_calculator.inner_solver
    inner_problem = spline_calculator.inner_problem

    return plot_splines_from_inner_result(
        inner_problem,
        inner_solver,
        inner_results,
        sim,
        observable_ids,
        axes=axes,
        rel_and_semiquant_obs_indices=rel_and_semiquant_obs_indices,
        **kwargs,
    )


def plot_splines_from_inner_result(
    inner_problem: "pypesto.hierarchical.spline_approximation.problem.SplineInnerProblem",
    inner_solver: "pypesto.hierarchical.spline_approximation.solver.SplineInnerSolver",
    results: list[dict],
    sim: list[np.ndarray],
    observable_ids=None,
    axes: np.ndarray | None = None,
    rel_and_semiquant_obs_indices: list[int] | None = None,
    **kwargs,
) -> np.ndarray:
    """Plot the estimated spline approximations from inner results.

    Parameters
    ----------
    inner_problem:
        The inner problem.
    inner_solver:
        The inner solver.
    results:
        The results from the inner solver.
    sim:
        The simulated model output.
    observable_ids:
        The ids of the observables.
    axes:
        Optional 2-D NumPy array of Axes to draw into. Required when
        ``rel_and_semiquant_obs_indices`` is set.
    rel_and_semiquant_obs_indices:
        Sorted indices of the relative and semi-quantitative observables in
        the AMICI model. Each observable is plotted on the subplot at the
        corresponding position in ``axes.flat``.
    kwargs:
        Additional arguments to pass to the plotting function.

    Returns
    -------
    axes:
        A 2-D NumPy array of matplotlib Axes.
    """

    if len(results) != len(inner_problem.groups):
        raise ValueError(
            "Number of results must be equal to number of groups of the inner subproblem."
        )

    # Get the number of groups
    n_groups = len(inner_problem.groups)
    semiquant_groups = list(inner_problem.groups.keys())

    if rel_and_semiquant_obs_indices is None:
        axes = _prepare_observable_mapping_axes(
            axes=axes,
            n_panels=n_groups,
            **kwargs,
        )
    else:
        if axes is None:
            raise ValueError(
                "Pass `axes` when `rel_and_semiquant_obs_indices` is set."
            )
        if not isinstance(axes, np.ndarray) or axes.ndim != 2:
            raise ValueError("`axes` must be a 2-D NumPy array.")
        if axes.size < len(rel_and_semiquant_obs_indices):
            raise ValueError(
                "The number of axes must be at least equal to the number "
                "of relative and semi-quantitative observables."
            )

    flat_axes = axes.flat

    # for each result and group, plot the inner solution
    for result, group in zip(results, inner_problem.groups, strict=True):
        if rel_and_semiquant_obs_indices is not None:
            ax_index = rel_and_semiquant_obs_indices.index(group - 1)
        else:
            ax_index = semiquant_groups.index(group)

        # For each group get the inner parameters and simulation
        s = result[SCIPY_X]

        # Utility matrix for the spline knot calculation
        lower_trian = np.tril(np.ones((len(s), len(s))))
        spline_knots = np.dot(lower_trian, s)

        measurements = inner_problem.groups[group][DATAPOINTS]
        simulation = extract_expdata_using_mask(
            expdata=sim, mask=inner_problem.groups[group][EXPDATA_MASK]
        )

        # For the simulation, get the spline bases
        (
            delta_c,
            spline_bases,
            n,
        ) = SemiquantInnerSolver._rescale_spline_bases(
            sim_all=simulation,
            N=len(spline_knots),
            K=len(simulation),
        )

        _plot_observable_mapping_measurements(
            flat_axes[ax_index],
            simulation,
            measurements,
        )
        flat_axes[ax_index].scatter(
            spline_bases,
            spline_knots,
            color="C2",
            s=30,
            alpha=0.9,
            linewidths=0.4,
            edgecolors="white",
            label="Spline knots",
            zorder=4,
        )
        flat_axes[ax_index].plot(
            spline_bases,
            spline_knots,
            linestyle="-",
            color="C2",
            linewidth=1.8,
            label="Spline function",
            zorder=2,
        )
        if inner_solver.options[REGULARIZE_SPLINE]:
            alpha_opt, beta_opt = _calculate_optimal_regularization(
                s=s,
                N=len(spline_knots),
                c=spline_bases,
            )
            flat_axes[ax_index].plot(
                spline_bases,
                alpha_opt * spline_bases + beta_opt,
                linestyle="--",
                color="C1",
                linewidth=1.5,
                label="Regularization line",
                zorder=1,
            )

        if observable_ids is not None:
            title = f"Observable {observable_ids[group - 1]}"
        else:
            title = f"Group {group}"

        _finalize_observable_mapping_axes(
            flat_axes[ax_index],
            title=title,
        )

    return axes


def _calculate_optimal_regularization(
    s: np.ndarray,
    N: int,
    c: np.ndarray,
):
    """Calculate the optimal linear regularization for the spline approximation.

    Parameters
    ----------
    s:
        The spline parameters.
    N:
        The number of inner parameters.
    c:
        The spline bases.

    Returns
    -------
    alpha_opt:
        The optimal slope of the linear function.
    beta_opt:
        The optimal offset of the linear function.
    """
    lower_trian = np.tril(np.ones((N, N)))
    xi = np.dot(lower_trian, s)

    # Calculate auxiliary values
    c_sum = np.sum(c)
    xi_sum = np.sum(xi)
    c_squares_sum = np.sum(c**2)
    c_dot_xi = np.dot(c, xi)
    # Calculate the optimal linear function offset
    if np.isclose(N * c_squares_sum - c_sum**2, 0):
        beta_opt = xi_sum / N
    else:
        beta_opt = (xi_sum * c_squares_sum - c_dot_xi * c_sum) / (
            N * c_squares_sum - c_sum**2
        )

    # If the offset is smaller than 0, we set it to 0
    if beta_opt < 0:
        beta_opt = 0

    # Calculate the slope of the optimal linear function
    alpha_opt = (c_dot_xi - beta_opt * c_sum) / c_squares_sum

    return alpha_opt, beta_opt


def _add_spline_mapped_simulations_to_model_fit(
    result: Result | Sequence[Result],
    pypesto_problem: Problem,
    start_index: int = 0,
    axes: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes | None:
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
            result.optimize_result.list[start_index]["x"],
            strict=True,
        )
    )
    x_dct.update(
        pypesto_problem.objective.calculator.necessary_par_dummy_values
    )
    # Get the needed objects from the pypesto problem.
    edatas = pypesto_problem.objective.edatas
    parameter_mapping = pypesto_problem.objective.parameter_mapping
    amici_model = pypesto_problem.objective.amici_model
    amici_solver = pypesto_problem.objective.amici_solver
    n_threads = pypesto_problem.objective.n_threads

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

    spline_calculator = None
    for calculator in pypesto_problem.objective.calculator.inner_calculators:
        if isinstance(calculator, SemiquantCalculator):
            spline_calculator = calculator
            break

    # Get the inner solver and problem.
    inner_solver = spline_calculator.inner_solver
    inner_problem = spline_calculator.inner_problem

    # Solve the inner problem.
    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    # Get the observable ids.
    observable_ids = amici_model.get_observable_ids()

    for inner_result, group in zip(
        inner_results, inner_problem.groups, strict=True
    ):
        observable_id = observable_ids[group - 1]
        # Get the ax for the current observable.
        ax = [
            ax
            for ax in axes.values()
            if observable_id in ax.legend().get_texts()[0].get_text().split()
        ][0]

        # Get the inner parameters and simulation.
        s = inner_result[SCIPY_X]
        simulation = inner_problem.groups[group][CURRENT_SIMULATION]

        # For the simulation, get the spline bases
        (
            delta_c,
            spline_bases,
            n,
        ) = SemiquantInnerSolver._rescale_spline_bases(
            sim_all=simulation,
            N=len(s),
            K=len(simulation),
        )
        # and the spline-mapped simulations.
        mapped_simulations = get_spline_mapped_simulations(
            s, simulation, len(s), delta_c, spline_bases, n
        )

        # Plot the spline-mapped simulations to the ax with same color
        # and timepoints as the lines which have 'simulation' in their label.
        plotted_index = 0
        for line in ax.lines:
            if "simulation" in line.get_label():
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
                    linestyle="dotted",
                    marker="^",
                )

        # Add linestyle='dotted' and marker='^' to the legend as black spline mapped simulations.
        ax.plot(
            [],
            [],
            color="black",
            linestyle="dotted",
            marker="^",
            label="Spline mapped simulation",
        )

    # Reset the legend.
    for ax in axes.values():
        ax.legend()

    return axes


def _obtain_regularization_for_start(
    pypesto_result: Result, start_index=0
) -> float | None:
    """Obtain the regularization for the start index.

    Calculates and returns the spline linear regularization
    term of the objective function for the start index.

    Parameters
    ----------
    pypesto_result:
        The pypesto result.
    start_index:
        The start index for which to calculate the regularization.

    Returns
    -------
    The regularization term of the objective function for the start index.
    """
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

    spline_calculator = None
    for (
        calculator
    ) in pypesto_result.problem.objective.calculator.inner_calculators:
        if isinstance(calculator, SemiquantCalculator):
            spline_calculator = calculator
            break

    # Get the inner solver and problem.
    inner_solver = spline_calculator.inner_solver
    inner_problem = spline_calculator.inner_problem

    inner_results = inner_solver.solve(inner_problem, sim, sigma)

    reg_term_sum = 0

    # for each result and group, plot the inner solution
    for result, group in zip(inner_results, inner_problem.groups, strict=True):
        # For each group get the inner parameters and simulation
        s = result[SCIPY_X]
        simulation = inner_problem.groups[group][CURRENT_SIMULATION]

        # For the simulation, get the spline bases
        (
            _,
            spline_bases,
            _,
        ) = SemiquantInnerSolver._rescale_spline_bases(
            sim_all=simulation,
            N=len(s),
            K=len(simulation),
        )

        if inner_solver.options[REGULARIZE_SPLINE]:
            reg_term = _calculate_regularization_for_group(
                s=s,
                N=len(s),
                c=spline_bases,
                regularization_factor=inner_solver.options[
                    "regularization_factor"
                ],
            )
            reg_term_sum += reg_term

    return reg_term_sum
