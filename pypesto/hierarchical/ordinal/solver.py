"""Definition of an optimal scaling solver class."""

import warnings

import numpy as np

from ...C import (
    C_MATRIX,
    CENSORED,
    INTERVAL_CONSTRAINTS,
    LB_INDICES,
    MAX,
    MAXMIN,
    MEASUREMENT_TYPE,
    METHOD,
    MIN_GAP,
    NUM_CATEGORIES,
    NUM_DATAPOINTS,
    NUM_INNER_PARAMS,
    ORDINAL,
    QUANTITATIVE_DATA,
    QUANTITATIVE_IXS,
    REDUCED,
    REPARAMETERIZED,
    SCIPY_FUN,
    SCIPY_SUCCESS,
    SCIPY_X,
    STANDARD,
    SURROGATE_DATA,
    UB_INDICES,
    W_DOT_MATRIX,
    W_MATRIX,
    InnerParameterType,
)
from ..base_solver import InnerSolver
from .parameter import OrdinalParameter
from .problem import OrdinalProblem

try:
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    ParameterMapping = None


class OrdinalInnerSolver(InnerSolver):
    """Solve the inner subproblem of the optimal scaling approach for ordinal data.

    Options
    -------
    method:
        The method to use for the inner optimization problem.
        Can be 'standard' or 'reduced'. The latter is more efficient.
    reparameterized:
        Whether to use reparameterized optimization.
    intervalConstraints:
        The type of interval constraints to use.
        Can be 'max' or 'maxmin'.
    minGap:
        The minimum gap between two consecutive categories.
    """

    def __init__(self, options: dict = None):
        """Construct."""
        self.options = {
            **self.get_default_options(),
            **(options or {}),
        }
        self.validate_options()

        self.x_guesses = None

    def validate_options(self):
        """Validate the current options dictionary."""
        if self.options[METHOD] not in [STANDARD, REDUCED]:
            raise ValueError(
                f"Inner solver method cannot be {self.options[METHOD]}. Please enter either {STANDARD} or {REDUCED}"
            )
        elif not isinstance(self.options[REPARAMETERIZED], bool):
            raise ValueError(
                f"Inner solver option 'reparameterized' has to be boolean, not {type(self.options[REPARAMETERIZED])}."
            )
        elif self.options[INTERVAL_CONSTRAINTS] not in [MAX, MAXMIN]:
            raise ValueError(
                f"Inner solver method cannot be {self.options[INTERVAL_CONSTRAINTS]}. Please enter either {MAX} or {MAXMIN}"
            )
        elif not isinstance(self.options[MIN_GAP], float):
            raise ValueError(
                f"Inner solver option 'reparameterized' has to be a float, not {type(self.options[MIN_GAP])}."
            )
        elif (
            self.options[METHOD] == STANDARD and self.options[REPARAMETERIZED]
        ):
            raise NotImplementedError(
                "Combining standard approach with "
                "reparameterization not implemented."
            )
        elif self.options[METHOD] == STANDARD:
            warnings.warn(
                "Standard approach is not recommended, as it is less efficient."
                "Please consider using the reduced approach instead.",
                stacklevel=2,
            )
        # Check for any other options
        for key in self.options:
            if key not in self.get_default_options():
                raise ValueError(
                    f"Unknown OptimalScalingInnerSolver option {key}."
                )

    def solve(
        self,
        problem: OrdinalProblem,
        sim: list[np.ndarray],
        sigma: list[np.ndarray],
    ) -> list:
        """Get results for every group (inner optimization problem).

        Parameters
        ----------
        problem:
            Optimal scaling inner problem.
        sim:
            Model simulations.
        sigma:
            Standard deviation of the noise.

        Returns
        -------
        List of optimization results of the inner subproblem.
        """
        optimal_surrogates = []
        for group in problem.get_groups_for_xs(InnerParameterType.ORDINAL):
            category_upper_bounds = problem.get_cat_ub_parameters_for_group(
                group
            )
            category_lower_bounds = problem.get_cat_lb_parameters_for_group(
                group
            )
            if problem.groups[group][MEASUREMENT_TYPE] == ORDINAL:
                surrogate_opt_results = optimize_surrogate_data_per_group(
                    category_upper_bounds=category_upper_bounds,
                    category_lower_bounds=category_lower_bounds,
                    sim=sim,
                    options=self.options,
                )
                save_inner_parameters_to_inner_problem(
                    category_upper_bounds=category_upper_bounds,
                    group=group,
                    problem=problem,
                    x_inner_opt=surrogate_opt_results,
                    sim=sim,
                    options=self.options,
                )
            elif problem.groups[group][MEASUREMENT_TYPE] == CENSORED:
                quantitative_data = problem.groups[group][QUANTITATIVE_DATA]
                quantitative_ixs = problem.groups[group][QUANTITATIVE_IXS]
                surrogate_opt_results = calculate_censored_obj(
                    category_upper_bounds=category_upper_bounds,
                    category_lower_bounds=category_lower_bounds,
                    sim=sim,
                    sigma=sigma,
                    quantitative_data=quantitative_data,
                    quantitative_ixs=quantitative_ixs,
                )

            optimal_surrogates.append(surrogate_opt_results)
        return optimal_surrogates

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """Calculate the inner objective function value.

        Calculates the inner objective function value from a list of inner
        optimization results returned from `OptimalScalingInnerSolver.solve`.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results of the inner subproblem.

        Returns
        -------
        Inner objective function value.
        """
        if False in [
            x_inner_opt[idx][SCIPY_SUCCESS] for idx in range(len(x_inner_opt))
        ]:
            obj = np.inf
            warnings.warn("Inner optimization failed.", stacklevel=2)
        else:
            obj = np.sum(
                [
                    x_inner_opt[idx][SCIPY_FUN]
                    for idx in range(len(x_inner_opt))
                ]
            )
        return obj

    def calculate_gradients(
        self,
        problem: OrdinalProblem,
        x_inner_opt: list[dict],
        sim: list[np.ndarray],
        sy: list[np.ndarray],
        sigma: list[np.ndarray],
        ssigma: list[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: list,
        par_sim_ids: list,
        par_edatas_indices: list,
        snllh: np.ndarray,
    ):
        """Calculate gradients of the inner objective function.

        Calculates gradients of the objective function with respect to outer
        (dynamical) parameters.

        Parameters
        ----------
        problem:
            Optimal scaling inner problem.
        x_inner_opt:
            List of optimization results of the inner subproblem.
        sim:
            Model simulations.
        sy:
            Model sensitivities.
        sigma:
            Model noise parameters.
        ssigma:
            Model sensitivity of noise parameters.
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        par_opt_ids:
            Ids of outer otimization parameters.
        par_sim_ids:
            Ids of outer simulation parameters, includes fixed parameters.
        par_edata_indices:
            Indices of parameters from `amici_model.getParameterIds()` that are needed for
            sensitivity calculation. Comes from `edata.plist` for each condition.
        snllh:
            A zero-initialized vector of the same length as ``par_opt_ids`` to store the
            gradients in. Will be modified in-place.

        Returns
        -------
        The gradients with respect to the outer parameters.
        """
        already_calculated = set()

        for condition_map_sim_var in [
            cond_par_map.map_sim_var for cond_par_map in parameter_mapping
        ]:
            # Iterate over outer optimization parameters.
            for par_sim, par_opt in condition_map_sim_var.items():
                if (
                    not isinstance(par_opt, str)
                    or par_opt in already_calculated
                ):
                    continue
                elif par_opt not in par_opt_ids:
                    continue
                else:
                    already_calculated.add(par_opt)

                par_opt_idx = par_opt_ids.index(par_opt)
                par_sim_idx = par_sim_ids.index(par_sim)
                par_edata_idx = [
                    (
                        par_edata_indices.index(par_sim_idx)
                        if par_sim_idx in par_edata_indices
                        else None
                    )
                    for par_edata_indices in par_edatas_indices
                ]

                grad = 0.0

                # Iterate over inner parameter groups.
                for idx, group in enumerate(
                    problem.get_groups_for_xs(InnerParameterType.ORDINAL)
                ):
                    if problem.groups[group][MEASUREMENT_TYPE] == CENSORED:
                        category_upper_bounds = (
                            problem.get_cat_ub_parameters_for_group(group)
                        )
                        category_lower_bounds = (
                            problem.get_cat_lb_parameters_for_group(group)
                        )
                        quantitative_data = problem.groups[group][
                            QUANTITATIVE_DATA
                        ]
                        quantitative_ixs = problem.groups[group][
                            QUANTITATIVE_IXS
                        ]

                        grad += calculate_censored_grad(
                            category_upper_bounds=category_upper_bounds,
                            category_lower_bounds=category_lower_bounds,
                            sim=sim,
                            sy=sy,
                            sigma=sigma,
                            ssigma=ssigma,
                            par_edata_idx=par_edata_idx,
                            quantitative_data=quantitative_data,
                            quantitative_ixs=quantitative_ixs,
                        )
                    elif problem.groups[group][MEASUREMENT_TYPE] == ORDINAL:
                        xs = problem.get_cat_ub_parameters_for_group(group)
                        xi = get_xi(
                            group, problem, x_inner_opt[idx], sim, self.options
                        )
                        sim_all = get_sim_all(xs, sim)
                        sy_all = get_sy_all(xs, sy, par_edata_idx)

                        problem.groups[group][W_MATRIX] = problem.get_w(
                            group, sim_all
                        )
                        problem.groups[group][W_DOT_MATRIX] = problem.get_wdot(
                            group, sim_all, sy_all
                        )

                        residual = np.block(
                            [
                                xi[: problem.groups[group][NUM_DATAPOINTS]]
                                - sim_all,
                                np.zeros(
                                    problem.groups[group][NUM_INNER_PARAMS]
                                    - problem.groups[group][NUM_DATAPOINTS]
                                ),
                            ]
                        )
                        dy_dtheta = get_dy_dtheta(group, problem, sy_all)

                        df_dtheta = residual.dot(
                            residual.dot(problem.groups[group][W_DOT_MATRIX])
                            - 2
                            * problem.groups[group][W_MATRIX].dot(dy_dtheta)
                        )
                        df_dxi = 2 * problem.groups[group][W_MATRIX].dot(
                            residual
                        )

                        if df_dxi.any():
                            dd_dtheta = problem.get_dd_dtheta(
                                group, xs, sim_all, sy_all
                            )
                            d = problem.get_d(
                                group, xs, sim_all, self.options[MIN_GAP]
                            )

                            mu = get_mu(group, problem, residual)

                            dxi_dtheta = calculate_dxi_dtheta(
                                group,
                                problem,
                                xi,
                                mu,
                                dy_dtheta,
                                residual,
                                d,
                                dd_dtheta,
                            )

                            grad += dxi_dtheta.dot(df_dxi) + df_dtheta
                        else:
                            grad += df_dtheta

                snllh[par_opt_idx] = grad
        return snllh

    @staticmethod
    def get_default_options() -> dict:
        """Return default options for solving the inner problem."""
        options = {
            METHOD: REDUCED,
            REPARAMETERIZED: True,
            INTERVAL_CONSTRAINTS: MAX,
            MIN_GAP: 1e-16,
        }
        return options


def calculate_dxi_dtheta(
    group: int,
    problem: OrdinalProblem,
    xi: np.ndarray,
    mu: np.ndarray,
    dy_dtheta: np.ndarray,
    residual: np.ndarray,
    d: np.ndarray,
    dd_dtheta: np.ndarray,
):
    """Calculate derivatives of inner parameters with respect to outer parameter.

    Parameters
    ----------
    group:
        Inner parameter group.
    problem:
        Optimal scaling inner problem.
    xi:
        Inner parameters: category bounds and surrogate data.
    mu:
        Lagrange multipliers of the inner optimization problem.
    dy_dtheta:
        Model sensitivities for group.
    residual:
        Residual for group.
    d:
        Vector of interval gap and range.
    dd_theta:
        Derivative of vector of interval gap and range.
    """
    from scipy.sparse import csc_matrix, linalg

    A = np.block(
        [
            [
                2 * problem.groups[group][W_MATRIX],
                problem.groups[group][C_MATRIX].transpose(),
            ],
            [
                (mu * problem.groups[group][C_MATRIX].transpose()).transpose(),
                np.diag(problem.groups[group][C_MATRIX].dot(xi) + d),
            ],
        ]
    )
    A_sp = csc_matrix(A)

    b = np.block(
        [
            2 * dy_dtheta.dot(problem.groups[group][W_MATRIX])
            - 2 * problem.groups[group][W_DOT_MATRIX].dot(residual),
            -mu * dd_dtheta,
        ]
    )

    dxi_dtheta = linalg.spsolve(A_sp, b)
    return dxi_dtheta[: problem.groups[group][NUM_INNER_PARAMS]]


def get_dy_dtheta(group: int, problem: OrdinalProblem, sy_all: np.ndarray):
    """Restructure sensitivities into a numpy matrix of right dimension."""
    return np.block(
        [sy_all, np.zeros(2 * problem.groups[group][NUM_CATEGORIES])]
    )


def get_mu(group: int, problem: OrdinalProblem, residual: np.ndarray):
    """Calculate Lagrange multipliers of the inner optimization problem.

    Parameters
    ----------
    group:
        Inner parameter group.
    problem:
        Optimal scaling inner problem.
    residual:
        Residual for group.
    """
    from scipy import linalg

    mu = linalg.lstsq(
        problem.groups[group][C_MATRIX].transpose(),
        -2 * residual.dot(problem.groups[group][W_MATRIX]),
        lapack_driver="gelsy",
    )
    return mu[0]


def get_xi(
    group: int,
    problem: OrdinalProblem,
    x_inner_opt: dict,
    sim: list[np.ndarray],
    options: dict,
):
    """Extract and calculate category bounds and surrogate data.

    Parameters
    ----------
    group:
        Inner parameter group.
    problem:
        Optimal scaling inner problem.
    x_inner_opt:
        Optimization results of the inner optimization subproblem.
    sim:
        Model simulation.
    options:
        Optimal scaling inner solver options.
    """
    xs = problem.get_cat_ub_parameters_for_group(group)
    interval_range, interval_gap = compute_interval_constraints(
        xs, sim, options
    )

    xi = np.zeros(problem.groups[group][NUM_INNER_PARAMS])
    surrogate_all, x_lower, x_upper = get_surrogate_all(
        xs, x_inner_opt[SCIPY_X], sim, interval_range, interval_gap, options
    )
    xi[: problem.groups[group][NUM_DATAPOINTS]] = surrogate_all.flatten()
    xi[problem.groups[group][LB_INDICES]] = x_lower
    xi[problem.groups[group][UB_INDICES]] = x_upper
    return xi


def optimize_surrogate_data_per_group(
    category_upper_bounds: list[OrdinalParameter],
    category_lower_bounds: list[OrdinalParameter],
    sim: list[np.ndarray],
    options: dict,
):
    """Run optimization for inner subproblem.

    Parameters
    ----------
    category_upper_bounds:
        Upper bound parameters of categories for this group.
    category_lower_bounds:
        Lower bound parameters of categories for this group.
    sim:
        Model simulations.
    options:
        Optimal scaling inner solver options.
    """
    from scipy.optimize import minimize

    interval_range, interval_gap = compute_interval_constraints(
        category_upper_bounds, sim, options
    )
    w = get_weight_for_surrogate(category_upper_bounds, sim)

    def obj_surr(x):
        return obj_surrogate_data(
            category_upper_bounds,
            x,
            sim,
            interval_gap,
            interval_range,
            w,
            options,
        )

    def grad_surr(x):
        return grad_surrogate_data(
            category_upper_bounds,
            x,
            sim,
            interval_gap,
            interval_range,
            w,
            options,
        )

    inner_options = get_inner_optimization_options(
        category_upper_bounds,
        category_lower_bounds,
        sim,
        interval_range,
        interval_gap,
        options,
    )
    try:
        results = minimize(obj_surr, jac=grad_surr, **inner_options)
    except ValueError:
        warnings.warn(
            "x0 violate bound constraints. Retrying with array of zeros.",
            stacklevel=2,
        )
        inner_options["x0"] = np.zeros(len(inner_options["x0"]))
        results = minimize(obj_surr, jac=grad_surr, **inner_options)

    return results


def get_inner_optimization_options(
    category_upper_bounds: list[OrdinalParameter],
    category_lower_bounds: list[OrdinalParameter],
    sim: list[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: dict,
) -> dict:
    """Return default options for scipy optimizer.

    Returns inner subproblem optimization options including startpoint
    and optimization bounds or constraints, dependent on solver method.

    Parameters
    ----------
    category_upper_bounds:
        Upper bound parameters of categories for this group.
    category_lower_bounds:
        Lower bound parameters of categories for this group.
    sim:
        Model simulations.
    interval_range:
        Minimal constrained category interval range.
    interval_gap:
        Minimal constrained gap between categories.
    options:
        Optimal scaling inner solver options.
    """
    from scipy.optimize import Bounds

    min_all, max_all = get_min_max(category_upper_bounds, sim)
    if options[METHOD] == REDUCED:
        last_opt_values = np.asarray([x.value for x in category_upper_bounds])
    elif options[METHOD] == STANDARD:
        last_opt_values = np.ravel(
            [
                np.asarray([x.value for x in category_lower_bounds]),
                np.asarray([x.value for x in category_upper_bounds]),
            ],
            "F",
        )

    if options[METHOD] == REDUCED:
        parameter_length = len(category_upper_bounds)
        if len(np.nonzero(last_opt_values)) > 0:
            x0 = last_opt_values
        else:
            x0 = np.linspace(
                np.max([min_all, interval_range]),
                max_all + (interval_range + interval_gap) * parameter_length,
                parameter_length,
            )
    elif options[METHOD] == STANDARD:
        parameter_length = 2 * len(category_upper_bounds)
        if len(np.nonzero(last_opt_values)) > 0:
            x0 = last_opt_values
        else:
            x0 = np.linspace(0, max_all + interval_range, parameter_length)
    else:
        raise NotImplementedError(
            f"Unknown optimal scaling method {options[METHOD]}. "
            f"Please use {STANDARD} or {REDUCED}."
        )

    if options[REPARAMETERIZED]:
        x0 = reparameterize_inner_parameters(
            x0, category_upper_bounds, interval_gap, interval_range
        )
        # If taking last_opt_values, it's possible that x0 is negative.
        # due to different simulations. Clip so bounds are satisfied.
        x0 = np.clip(x0, 0, None)

        bounds = Bounds(
            [0.0] * parameter_length,
            [max_all + (interval_range + interval_gap) * parameter_length]
            * parameter_length,
        )
        inner_options = {
            "x0": x0,
            "method": "L-BFGS-B",
            "options": {"maxiter": 2000, "ftol": 1e-10},
            "bounds": bounds,
        }
    else:
        constraints = get_constraints_for_optimization(
            category_upper_bounds, sim, options
        )

        inner_options = {
            "x0": x0,
            "method": "SLSQP",
            "options": {"maxiter": 2000, "ftol": 1e-10, "disp": None},
            "constraints": constraints,
        }
    return inner_options


def get_min_max(
    inner_parameters: list[OrdinalParameter], sim: list[np.ndarray]
) -> tuple[float, float]:
    """Return minimal and maximal simulation value."""
    sim_all = get_sim_all(inner_parameters, sim)

    min_all = np.min(sim_all)
    max_all = np.max(sim_all)

    return min_all, max_all


def get_sy_all(
    inner_parameters: list[OrdinalParameter],
    sy: list[np.ndarray],
    par_edata_idx: list,
):
    """Return model sensitivities for inner parameters and outer parameter index."""
    sy_all = []
    for inner_parameter in inner_parameters:
        for sy_i, mask_i, edata_idx in zip(
            sy, inner_parameter.ixs, par_edata_idx
        ):
            if edata_idx is not None:
                sim_sy = sy_i[:, edata_idx, :][mask_i]
            else:
                sim_sy = np.full(sy_i[:, 0, :][mask_i].shape, 0)
            for sim_sy_i in sim_sy:
                sy_all.append(sim_sy_i)
    return np.array(sy_all)


def get_sim_all(inner_parameters, sim: list[np.ndarray]) -> list:
    """Return model simulations for inner parameters."""
    sim_all = []
    for inner_parameter in inner_parameters:
        for sim_i, mask_i in zip(sim, inner_parameter.ixs):
            sim_x = sim_i[mask_i]
            for sim_x_i in sim_x:
                sim_all.append(sim_x_i)
    return sim_all


def get_surrogate_all(
    inner_parameters: list[OrdinalParameter],
    optimal_scaling_bounds: list,
    sim: list[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: dict,
):
    """Return surrogate data, lower and upper category bounds."""
    if options[REPARAMETERIZED]:
        optimal_scaling_bounds = undo_inner_parameter_reparameterization(
            optimal_scaling_bounds,
            inner_parameters,
            interval_gap,
            interval_range,
        )
    surrogate_all = []
    x_lower_all = []
    x_upper_all = []
    for inner_parameter in inner_parameters:
        upper_bound, lower_bound = get_bounds_for_category(
            inner_parameter, optimal_scaling_bounds, interval_gap, options
        )
        for sim_i, mask_i in zip(sim, inner_parameter.ixs):
            y_sim = sim_i[mask_i]
            for y_sim_i in y_sim:
                if lower_bound > y_sim_i:
                    y_surrogate = lower_bound
                elif y_sim_i > upper_bound:
                    y_surrogate = upper_bound
                elif lower_bound <= y_sim_i <= upper_bound:
                    y_surrogate = y_sim_i
                else:
                    continue
                surrogate_all.append(y_surrogate)
        x_lower_all.append(lower_bound)
        x_upper_all.append(upper_bound)
    return (
        np.array(surrogate_all),
        np.array(x_lower_all),
        np.array(x_upper_all),
    )


def get_weight_for_surrogate(
    inner_parameters: list[OrdinalParameter], sim: list[np.ndarray]
) -> float:
    """Calculate weights for objective function."""
    sim_x_all = get_sim_all(inner_parameters, sim)
    eps = 1e-8

    w = np.sum(np.abs(sim_x_all)) + eps

    return w


def compute_interval_constraints(
    inner_parameters: list[OrdinalParameter],
    sim: list[np.ndarray],
    options: dict,
) -> tuple[float, float]:
    """Compute minimal interval range and gap."""
    # compute constraints on interval size and interval gap size
    # similar to Pargett et al. (2014)
    if MIN_GAP not in options:
        eps = 1e-16
    else:
        eps = options[MIN_GAP]

    min_simulation, max_simulation = get_min_max(inner_parameters, sim)

    if options[INTERVAL_CONSTRAINTS] == MAXMIN:
        interval_range = (max_simulation - min_simulation) / (
            2 * len(inner_parameters) + 1
        )
        interval_gap = (max_simulation - min_simulation) / (
            4 * (len(inner_parameters) - 1) + 1
        )
    elif options[INTERVAL_CONSTRAINTS] == MAX:
        interval_range = max_simulation / (2 * len(inner_parameters) + 1)
        interval_gap = max_simulation / (4 * (len(inner_parameters) - 1) + 1)
    else:
        raise NotImplementedError(
            f"intervalConstraints = "
            f"{options[INTERVAL_CONSTRAINTS]} not implemented. "
            f"Please use {MAX} or {MAXMIN}."
        )
    return interval_range, interval_gap + eps


def reparameterize_inner_parameters(
    original_inner_parameter_values: np.ndarray,
    inner_parameters: list[OrdinalParameter],
    interval_gap: float,
    interval_range: float,
) -> np.ndarray:
    """Transform original inner parameters to reparameterized inner parameters."""
    reparameterized_inner_parameter_values = np.full(
        shape=(np.shape(original_inner_parameter_values)), fill_value=np.nan
    )
    for inner_parameter in inner_parameters:
        inner_parameter_category = int(inner_parameter.category)
        if inner_parameter_category == 1:
            reparameterized_inner_parameter_values[
                inner_parameter_category - 1
            ] = (
                original_inner_parameter_values[inner_parameter_category - 1]
                - interval_range
            )
        else:
            reparameterized_inner_parameter_values[
                inner_parameter_category - 1
            ] = (
                original_inner_parameter_values[inner_parameter_category - 1]
                - original_inner_parameter_values[inner_parameter_category - 2]
                - interval_gap
                - interval_range
            )

    return reparameterized_inner_parameter_values


def undo_inner_parameter_reparameterization(
    reparameterized_inner_parameter_values: np.ndarray,
    inner_parameters: list[OrdinalParameter],
    interval_gap: float,
    interval_range: float,
) -> np.ndarray:
    """Transform reparameterized inner parameters to original inner parameters."""
    original_inner_parameter_values = np.full(
        shape=(np.shape(reparameterized_inner_parameter_values)),
        fill_value=np.nan,
    )
    for inner_parameter in inner_parameters:
        inner_parameter_category = int(inner_parameter.category)
        if inner_parameter_category == 1:
            original_inner_parameter_values[inner_parameter_category - 1] = (
                interval_range
                + reparameterized_inner_parameter_values[
                    inner_parameter_category - 1
                ]
            )
        else:
            original_inner_parameter_values[inner_parameter_category - 1] = (
                reparameterized_inner_parameter_values[
                    inner_parameter_category - 1
                ]
                + interval_gap
                + interval_range
                + original_inner_parameter_values[inner_parameter_category - 2]
            )
    return original_inner_parameter_values


def obj_surrogate_data(
    xs: list[OrdinalParameter],
    optimal_scaling_bounds: np.ndarray,
    sim: list[np.ndarray],
    interval_gap: float,
    interval_range: float,
    w: float,
    options: dict,
) -> float:
    """Compute optimal scaling objective function."""
    obj = 0.0
    if options[REPARAMETERIZED]:
        optimal_scaling_bounds = undo_inner_parameter_reparameterization(
            optimal_scaling_bounds, xs, interval_gap, interval_range
        )

    for x in xs:
        x_upper, x_lower = get_bounds_for_category(
            x, optimal_scaling_bounds, interval_gap, options
        )
        for sim_i, mask_i in zip(sim, x.ixs):
            y_sim = sim_i[mask_i]
            for y_sim_i in y_sim:
                if x_lower > y_sim_i:
                    y_surrogate = x_lower
                elif y_sim_i > x_upper:
                    y_surrogate = x_upper
                elif x_lower <= y_sim_i <= x_upper:
                    y_surrogate = y_sim_i
                else:
                    continue
                obj += (y_surrogate - y_sim_i) ** 2
    obj = np.divide(obj, w)
    return obj


def grad_surrogate_data(
    xs: list[OrdinalParameter],
    optimal_scaling_bounds: np.ndarray,
    sim: list[np.ndarray],
    interval_gap: float,
    interval_range: float,
    w: float,
    options: dict,
) -> float:
    """Compute optimal scaling objective function."""
    grad = np.zeros(len(optimal_scaling_bounds))
    if options[REPARAMETERIZED]:
        optimal_scaling_bounds = undo_inner_parameter_reparameterization(
            optimal_scaling_bounds, xs, interval_gap, interval_range
        )

    if options[METHOD] == STANDARD:
        for x in xs:
            x_category = int(x.category)
            x_lower = optimal_scaling_bounds[2 * x_category - 2]
            x_upper = optimal_scaling_bounds[2 * x_category - 1]

            for sim_i, mask_i in zip(sim, x.ixs):
                y_sim = sim_i[mask_i]
                for y_sim_i in y_sim:
                    if x_lower > y_sim_i:
                        y_surrogate = x_lower
                        grad[2 * x_category - 2] += 2 * (y_surrogate - y_sim_i)
                    elif y_sim_i > x_upper:
                        y_surrogate = x_upper
                        grad[2 * x_category - 1] += 2 * (y_surrogate - y_sim_i)
                    else:
                        continue
    elif options[METHOD] == REDUCED:
        for x in xs:
            x_category = int(x.category)
            x_upper = optimal_scaling_bounds[x_category - 1]
            if x_category == 1:
                x_lower = 0.0
            else:
                x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap

            for sim_i, mask_i in zip(sim, x.ixs):
                y_sim = sim_i[mask_i]
                for y_sim_i in y_sim:
                    if x_lower > y_sim_i and x_category != 1:
                        y_surrogate = x_lower
                        grad[x_category - 2] += 2 * (y_surrogate - y_sim_i)
                    elif y_sim_i > x_upper:
                        y_surrogate = x_upper
                        grad[x_category - 1] += 2 * (y_surrogate - y_sim_i)
                    else:
                        continue

    if options[REPARAMETERIZED]:
        grad = reparameterize_gradient(
            grad,
            xs,
        )
    grad = np.divide(grad, w)
    return grad


def reparameterize_gradient(
    grad: np.ndarray,
    xs: list[OrdinalParameter],
) -> np.ndarray:
    """Transform gradient to reparameterized gradient."""
    reparameterized_grad = np.full(
        shape=(np.shape(grad)),
        fill_value=np.nan,
    )
    for inner_parameter in xs:
        inner_parameter_category = int(inner_parameter.category)
        reparameterized_grad[inner_parameter_category - 1] = np.sum(
            grad[inner_parameter_category - 1 :]
        )
    return reparameterized_grad


def get_bounds_for_category(
    x: OrdinalParameter,
    optimal_scaling_bounds: np.ndarray,
    interval_gap: float,
    options: dict,
) -> tuple[float, float]:
    """Return upper and lower bound for a specific category x."""
    x_category = int(x.category)

    if options[METHOD] == REDUCED:
        x_upper = optimal_scaling_bounds[x_category - 1]
        if x_category == 1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        else:
            raise ValueError("Category value needs to be larger than 0.")
    elif options[METHOD] == STANDARD:
        x_lower = optimal_scaling_bounds[2 * x_category - 2]
        x_upper = optimal_scaling_bounds[2 * x_category - 1]
    else:
        raise NotImplementedError(
            f"Unknown optimal scaling method {options[METHOD]}. "
            f"Please use {REDUCED} or {STANDARD}."
        )
    return x_upper, x_lower


def get_constraints_for_optimization(
    xs: list[OrdinalParameter], sim: list[np.ndarray], options: dict
) -> dict:
    """Return constraints for inner optimization."""
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(
        xs, sim, options
    )
    if options[METHOD] == REDUCED:
        a = np.diag(-np.ones(num_categories), -1) + np.diag(
            np.ones(num_categories + 1)
        )
        a = a[:-1, :-1]
        b = np.empty((num_categories,))
        b[0] = interval_range
        b[1:] = interval_range + interval_gap
    elif options[METHOD] == STANDARD:
        a = np.diag(-np.ones(2 * num_categories), -1) + np.diag(
            np.ones(2 * num_categories + 1)
        )
        a = a[:-1, :-1]
        b = np.empty((2 * num_categories,))
        b[0] = 0
        b[1::2] = interval_range
        b[2::2] = interval_gap
    ineq_cons = {"type": "ineq", "fun": lambda x: a.dot(x) - b}

    return ineq_cons


def save_inner_parameters_to_inner_problem(
    category_upper_bounds: list[OrdinalParameter],
    group: int,
    problem: OrdinalProblem,
    x_inner_opt: dict,
    sim: list[np.ndarray],
    options: dict,
) -> None:
    """Save inner parameter values to the inner subproblem."""
    interval_range, interval_gap = compute_interval_constraints(
        category_upper_bounds, sim, options
    )

    surrogate_all, x_lower, x_upper = get_surrogate_all(
        category_upper_bounds,
        x_inner_opt[SCIPY_X],
        sim,
        interval_range,
        interval_gap,
        options,
    )
    problem.groups[group][SURROGATE_DATA] = surrogate_all.flatten()

    for inner_parameter in problem.get_cat_ub_parameters_for_group(group):
        inner_parameter.value = x_upper[inner_parameter.category - 1]
    for inner_parameter in problem.get_cat_lb_parameters_for_group(group):
        inner_parameter.value = x_lower[inner_parameter.category - 1]


def calculate_censored_obj(
    category_upper_bounds: list[OrdinalParameter],
    category_lower_bounds: list[OrdinalParameter],
    sim: list[np.ndarray],
    sigma: list[np.ndarray],
    quantitative_data: np.ndarray,
    quantitative_ixs: list[np.ndarray],
) -> dict:
    """Calculate objective function for a group with censored data.

    Parameters
    ----------
    category_upper_bounds:
        The upper bounds for the categories.
    category_lower_bounds:
        The lower bounds for the categories.
    sim:
        The model simulation.
    sigma:
        The noise parameters from AMICI.
    observable_ids:
        The observable ids from the model.

    Returns
    -------
    Dictionary with the objective function value, dummy success
    and censoring category bounds.
    """
    cat_lb_values = np.array([x.value for x in category_lower_bounds])
    cat_ub_values = np.array([x.value for x in category_upper_bounds])

    obj = 0
    # Calculate the objective function for censored data.
    for cat_lb, cat_ub, cat_ub_par in zip(
        cat_lb_values,
        cat_ub_values,
        category_upper_bounds,
    ):
        for sim_i, sigma_i, mask_i in zip(sim, sigma, cat_ub_par.ixs):
            y_sim = sim_i[mask_i]
            if len(y_sim) == 0:
                continue
            sigma_for_observable = sigma_i[mask_i][0]
            for y_sim_i in y_sim:
                if cat_lb > y_sim_i:
                    y_surrogate = cat_lb
                elif y_sim_i > cat_ub:
                    y_surrogate = cat_ub
                else:
                    y_surrogate = y_sim_i
                obj += 0.5 * (
                    np.log(2 * np.pi * sigma_for_observable**2)
                    + (y_surrogate - y_sim_i) ** 2 / sigma_for_observable**2
                )

    # Gather the simulation and sigma values for the quantitative data.
    quantitative_sim = np.concatenate(
        [sim_i[mask_i] for sim_i, mask_i in zip(sim, quantitative_ixs)]
    )
    quantitative_sigma = np.concatenate(
        [sigma_i[mask_i] for sigma_i, mask_i in zip(sigma, quantitative_ixs)]
    )

    # Calculate the objective function for uncensored, quantitative data.
    obj += 0.5 * np.nansum(
        np.log(2 * np.pi * quantitative_sigma**2)
        + (quantitative_data - quantitative_sim) ** 2 / quantitative_sigma**2
    )

    return_dictionary = {
        SCIPY_SUCCESS: True,
        SCIPY_FUN: obj,
        SCIPY_X: np.ravel([cat_lb_values, cat_ub_values], order="F"),
    }
    return return_dictionary


def calculate_censored_grad(
    category_upper_bounds: list[OrdinalParameter],
    category_lower_bounds: list[OrdinalParameter],
    sim: list[np.ndarray],
    sy: np.ndarray,
    sigma: list[np.ndarray],
    ssigma: np.ndarray,
    par_edata_idx: list,
    quantitative_data: np.ndarray,
    quantitative_ixs: list[np.ndarray],
):
    """Calculate gradient for a group with censored data with respect to an outer parameter.

    Parameters
    ----------
    category_upper_bounds:
        The upper bounds for the categories.
    category_lower_bounds:
        The lower bounds for the categories.
    sim:
        The model simulation.
    sy_all:
        The sensitivities of the simulation.
    sigma:
        The noise parameters from AMICI.
    ssigma_all:
        The sensitivities of the noise parameters.

    Returns
    -------
    Gradient.
    """
    cat_lb_values = np.array([x.value for x in category_lower_bounds])
    cat_ub_values = np.array([x.value for x in category_upper_bounds])

    surrogate_all = []
    sim_all = []
    sigma_all = []
    sy_all = get_sy_all(category_upper_bounds, sy, par_edata_idx)
    ssigma_all = get_sy_all(category_upper_bounds, ssigma, par_edata_idx)

    # Gather the surrogate data, the simulation data
    # and the noise parameter arrays across categories.
    for cat_lb, cat_ub, cat_ub_par in zip(
        cat_lb_values,
        cat_ub_values,
        category_upper_bounds,
    ):
        for sim_i, sigma_i, mask_i in zip(sim, sigma, cat_ub_par.ixs):
            y_sim = sim_i[mask_i]
            if len(y_sim) == 0:
                continue
            sigma_for_observable = sigma_i[mask_i][0]
            for y_sim_i in y_sim:
                if cat_lb > y_sim_i:
                    y_surrogate = cat_lb
                elif y_sim_i > cat_ub:
                    y_surrogate = cat_ub
                else:
                    y_surrogate = y_sim_i
                sim_all.append(y_sim_i)
                sigma_all.append(sigma_for_observable)
                surrogate_all.append(y_surrogate)

    sim_all = np.array(sim_all)
    sigma_all = np.array(sigma_all)
    surrogate_all = np.array(surrogate_all)

    # Calculate the negative log likelihood gradient for censored data.
    gradient = np.nansum(
        (
            np.full(len(sim_all), 1)
            - (surrogate_all - sim_all) ** 2 / sigma_all**2
        )
        * ssigma_all
        / sigma_all
    ) - np.nansum((surrogate_all - sim_all) * sy_all / sigma_all**2)

    # Gather the simulation, sigma values and sensitivities for the quantitative data.
    quantitative_sim = np.concatenate(
        [sim_i[mask_i] for sim_i, mask_i in zip(sim, quantitative_ixs)]
    )
    quantitative_sigma = np.concatenate(
        [sigma_i[mask_i] for sigma_i, mask_i in zip(sigma, quantitative_ixs)]
    )
    quantitative_sy = np.concatenate(
        [
            (
                sy_i[:, edata_idx, :][mask_i]
                if edata_idx is not None
                else np.full(sy_i[:, 0, :][mask_i].shape, 0)
            )
            for sy_i, mask_i, edata_idx in zip(
                sy, quantitative_ixs, par_edata_idx
            )
        ]
    )
    quantitative_ssigma = np.concatenate(
        [
            (
                ssigma_i[:, edata_idx, :][mask_i]
                if edata_idx is not None
                else np.full(ssigma_i[:, 0, :][mask_i].shape, 0)
            )
            for ssigma_i, mask_i, edata_idx in zip(
                ssigma, quantitative_ixs, par_edata_idx
            )
        ]
    )

    # Calculate the negative log likelihood gradient for uncensored, quantitative data.
    gradient += np.nansum(
        (
            np.full(len(quantitative_sim), 1)
            - (quantitative_data - quantitative_sim) ** 2
            / quantitative_sigma**2
        )
        * quantitative_ssigma
        / quantitative_sigma
    ) - np.nansum(
        (quantitative_data - quantitative_sim)
        * quantitative_sy
        / quantitative_sigma**2
    )

    return gradient
