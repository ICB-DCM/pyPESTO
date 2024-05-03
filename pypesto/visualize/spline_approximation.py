import warnings
from collections.abc import Sequence
from typing import Optional, Union

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
    REGULARIZE_SPLINE,
    SCIPY_X,
    SPLINE_KNOTS,
)
from ..problem import Problem
from ..result import Result

try:
    import amici
    from amici.petab.conditions import fill_in_parameters

    from ..hierarchical import InnerCalculatorCollector
    from ..hierarchical.semiquantitative.calculator import SemiquantCalculator
    from ..hierarchical.semiquantitative.solver import (
        SemiquantInnerSolver,
        _calculate_regularization_for_group,
        extract_expdata_using_mask,
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
    # Check the calculator is the InnerCalculatorCollector.
    if not isinstance(
        pypesto_result.problem.objective.calculator, InnerCalculatorCollector
    ):
        raise ValueError(
            "The calculator must be an instance of the InnerCalculatorCollector."
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
    observable_ids = amici_model.getObservableIds()

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
        **kwargs,
    )


def plot_splines_from_inner_result(
    inner_problem: "pypesto.hierarchical.spline_approximation.problem.SplineInnerProblem",
    inner_solver: "pypesto.hierarchical.spline_approximation.solver.SplineInnerSolver",
    results: list[dict],
    sim: list[np.ndarray],
    observable_ids=None,
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
    sim:
        The simulated model output.
    observable_ids:
        The ids of the observables.
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
        mapped_simulations = get_spline_mapped_simulations(
            s, simulation, len(spline_knots), delta_c, spline_bases, n
        )

        axs[group_idx].plot(
            simulation, measurements, "bs", label="Measurements"
        )
        axs[group_idx].plot(
            spline_bases, spline_knots, "g.", label="Spline knots"
        )
        axs[group_idx].plot(
            spline_bases,
            spline_knots,
            linestyle="-",
            color="g",
            label="Spline function",
        )
        if inner_solver.options[REGULARIZE_SPLINE]:
            alpha_opt, beta_opt = _calculate_optimal_regularization(
                s=s,
                N=len(spline_knots),
                c=spline_bases,
            )
            axs[group_idx].plot(
                spline_bases,
                alpha_opt * spline_bases + beta_opt,
                linestyle="--",
                color="orange",
                label="Regularization line",
            )

        axs[group_idx].plot(
            simulation, mapped_simulations, "r^", label="Mapped simulation"
        )
        axs[group_idx].legend()
        if observable_ids is not None:
            axs[group_idx].set_title(f"{observable_ids[group-1]}")
        else:
            axs[group_idx].set_title(f"Group {group}")

        axs[group_idx].set_xlabel("Model output")
        axs[group_idx].set_ylabel("Measurements")

    for ax in axs[len(results) :]:
        ax.remove()

    return fig, axs


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
            result.optimize_result.list[start_index]["x"],
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
) -> Optional[float]:
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
    for result, group in zip(inner_results, inner_problem.groups):
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
