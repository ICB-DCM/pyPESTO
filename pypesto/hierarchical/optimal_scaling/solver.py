"""Definition of an optimal scaling solver class."""
import warnings
from typing import Dict, List, Tuple

import numpy as np

from ...C import MAX, MAXMIN, REDUCED, STANDARD, InnerParameterType
from ..solver import InnerSolver
from .parameter import OptimalScalingParameter
from .problem import OptimalScalingProblem

try:
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass


class OptimalScalingInnerSolver(InnerSolver):
    """Solve the inner subproblem of the optimal scaling approach for ordinal data."""

    def __init__(self, options: Dict = None):
        """Construct the optimal scaling inner solver.

        Parameters
        ----------
        options:
            Inner solver options. If not given will take default ones.
            Needs to contain method ('standard' or 'reduced'), reparameterized (Boolean),
            intervalConstraints ('max' or 'maxmin'), and minGap (float).
        """
        self.options = options

        if not self.options:
            self.options = self.get_default_options()
        else:
            self.validate_options()

        self.x_guesses = None

    def validate_options(self):
        """Validate the current options dictionary."""
        if not all(
            option in self.options
            for option in [
                'method',
                'reparameterized',
                'intervalConstraints',
                'minGap',
            ]
        ):
            default_options = self.get_default_options()
            missing_options = [
                option
                for option in [
                    'method',
                    'reparameterized',
                    'intervalConstraints',
                    'minGap',
                ]
                if option not in self.options
            ]

            warnings.warn(
                f"Adding default inner solver options for {missing_options}."
            )

            for option in missing_options:
                self.options[option] = default_options[option]

        elif self.options['method'] not in [STANDARD, REDUCED]:
            raise ValueError(
                f"Inner solver method cannot be {self.options['method']}. Please enter either {STANDARD} or {REDUCED}"
            )
        elif type(self.options['reparameterized']) is not bool:
            raise ValueError(
                f"Inner solver option 'reparameterized' has to be boolean, not {type(self.options['reparameterized'])}."
            )
        elif self.options['intervalConstraints'] not in [MAX, MAXMIN]:
            raise ValueError(
                f"Inner solver method cannot be {self.options['intervalConstraints']}. Please enter either {MAX} or {MAXMIN}"
            )
        elif type(self.options['minGap']) is not float:
            raise ValueError(
                f"Inner solver option 'reparameterized' has to be a float, not {type(self.options['minGap'])}."
            )
        elif (
            self.options['method'] == STANDARD
            and self.options['reparameterized']
        ):
            raise NotImplementedError(
                'Combining standard approach with '
                'reparameterization not implemented.'
            )

    def solve(
        self,
        problem: OptimalScalingProblem,
        sim: List[np.ndarray],
    ) -> list:
        """Get results for every group (inner optimization problem).

        Parameters
        ----------
        problem:
            Optimal scaling inner problem.
        sim:
            Model simulations.

        Returns
        -------
        List of optimization results of the inner subproblem.
        """
        optimal_surrogates = []
        for group in problem.get_groups_for_xs(
            InnerParameterType.OPTIMAL_SCALING
        ):
            category_upper_bounds = problem.get_cat_ub_parameters_for_group(
                group
            )
            category_lower_bounds = problem.get_cat_lb_parameters_for_group(
                group
            )
            surrogate_opt_results = optimize_surrogate_data_per_group(
                category_upper_bounds=category_upper_bounds,
                category_lower_bounds=category_lower_bounds,
                sim=sim,
                options=self.options,
            )
            save_inner_parameters_to_inner_problem(
                category_upper_bounds=category_upper_bounds,
                gr=group,
                problem=problem,
                x_inner_opt=surrogate_opt_results,
                sim=sim,
                options=self.options,
            )
            optimal_surrogates.append(surrogate_opt_results)
        return optimal_surrogates

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """Calculate the inner objective function value.

        Calculates the inner objective function value from a list of inner
        optimization results returned from compute_optimal_surrogate_data.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results of the inner subproblem.

        Returns
        -------
        Inner objective function value.
        """
        if False in [
            x_inner_opt[idx]['success'] for idx in range(len(x_inner_opt))
        ]:
            obj = np.inf
            warnings.warn("Inner optimization failed.")
        else:
            obj = np.sum(
                [x_inner_opt[idx]['fun'] for idx in range(len(x_inner_opt))]
            )
        return obj

    def calculate_gradients(
        self,
        problem: OptimalScalingProblem,
        x_inner_opt: List[Dict],
        sim: List[np.ndarray],
        sy: List[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: List,
        par_sim_ids: List,
        snllh: Dict,
    ):
        """Calculate gradients of the inner objective function.

        Calculates gradients of the objective function with respect to outer
        parameters.

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
        parameter_mapping:
            Mapping of optimization to simulation parameters.
        par_opt_ids:
            Ids of outer otimization parameters.
        par_sim_ids:
            Ids of outer simulation parameters, includes fixed parameters.
        snllh:
            Empty dictionary with optimization parameters as keys.

        Returns
        -------
        Filled in snllh dictionary with objective function gradients.
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
                else:
                    already_calculated.add(par_opt)
                par_sim_idx = par_sim_ids.index(par_sim)
                par_opt_idx = par_opt_ids.index(par_opt)
                grad = 0.0

                # Iterate over inner parameter groups.
                for idx, group in enumerate(
                    problem.get_groups_for_xs(
                        InnerParameterType.OPTIMAL_SCALING
                    )
                ):
                    xs = problem.get_cat_ub_parameters_for_group(group)
                    xi = get_xi(
                        group, problem, x_inner_opt[idx], sim, self.options
                    )
                    sim_all = get_sim_all(xs, sim)
                    sy_all = get_sy_all(xs, sy, par_sim_idx)

                    problem.groups[group]['W'] = problem.get_w(group, sim_all)
                    problem.groups[group]['Wdot'] = problem.get_wdot(
                        group, sim_all, sy_all
                    )

                    residual = np.block(
                        [
                            xi[: problem.groups[group]['num_datapoints']]
                            - sim_all,
                            np.zeros(
                                problem.groups[group]['num_inner_params']
                                - problem.groups[group]['num_datapoints']
                            ),
                        ]
                    )
                    dy_dtheta = get_dy_dtheta(group, problem, sy_all)

                    df_dtheta = residual.dot(
                        residual.dot(problem.groups[group]['Wdot'])
                        - 2 * problem.groups[group]['W'].dot(dy_dtheta)
                    )
                    df_dxi = 2 * problem.groups[group]['W'].dot(residual)

                    if df_dxi.any():
                        dd_dtheta = problem.get_dd_dtheta(
                            group, xs, sim_all, sy_all
                        )
                        d = problem.get_d(
                            group, xs, sim_all, self.options['minGap']
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
    def get_default_options() -> Dict:
        """Return default options for solving the inner problem."""
        options = {
            'method': REDUCED,
            'reparameterized': True,
            'intervalConstraints': MAX,
            'minGap': 1e-16,
        }
        return options


def calculate_dxi_dtheta(
    group,
    problem: OptimalScalingProblem,
    xi,
    mu,
    dy_dtheta,
    residual,
    d,
    dd_dtheta,
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
                2 * problem.groups[group]['W'],
                problem.groups[group]['C'].transpose(),
            ],
            [
                (mu * problem.groups[group]['C'].transpose()).transpose(),
                np.diag(problem.groups[group]['C'].dot(xi) + d),
            ],
        ]
    )
    A_sp = csc_matrix(A)

    b = np.block(
        [
            2 * dy_dtheta.dot(problem.groups[group]['W'])
            - 2 * problem.groups[group]['Wdot'].dot(residual),
            -mu * dd_dtheta,
        ]
    )

    dxi_dtheta = linalg.spsolve(A_sp, b)
    return dxi_dtheta[: problem.groups[group]['num_inner_params']]


def get_dy_dtheta(gr, problem: OptimalScalingProblem, sy_all):
    """Restructure sensitivities into a numpy matrix of right dimension."""
    return np.block(
        [sy_all, np.zeros(2 * problem.groups[gr]['num_categories'])]
    )


def get_mu(group, problem: OptimalScalingProblem, residual):
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
        problem.groups[group]['C'].transpose(),
        -2 * residual.dot(problem.groups[group]['W']),
        lapack_driver='gelsy',
    )
    return mu[0]


def get_xi(
    gr,
    problem: OptimalScalingProblem,
    x_inner_opt: Dict,
    sim: List[np.ndarray],
    options: Dict,
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
    xs = problem.get_cat_ub_parameters_for_group(gr)
    interval_range, interval_gap = compute_interval_constraints(
        xs, sim, options
    )

    xi = np.zeros(problem.groups[gr]['num_inner_params'])
    surrogate_all, x_lower, x_upper = get_surrogate_all(
        xs, x_inner_opt['x'], sim, interval_range, interval_gap, options
    )
    xi[: problem.groups[gr]['num_datapoints']] = surrogate_all.flatten()
    xi[problem.groups[gr]['lb_indices']] = x_lower
    xi[problem.groups[gr]['ub_indices']] = x_upper
    return xi


def optimize_surrogate_data_per_group(
    category_upper_bounds: List[OptimalScalingParameter],
    category_lower_bounds: List[OptimalScalingParameter],
    sim: List[np.ndarray],
    options: Dict,
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

    inner_options = get_inner_optimization_options(
        category_upper_bounds,
        category_lower_bounds,
        sim,
        interval_range,
        interval_gap,
        options,
    )
    try:
        results = minimize(obj_surr, **inner_options)
    except BaseException:
        warnings.warn(
            "x0 violate bound constraints. Retrying with array of zeros."
        )
        inner_options['x0'] = np.zeros(len(inner_options['x0']))
        results = minimize(obj_surr, **inner_options)

    return results


def get_inner_optimization_options(
    category_upper_bounds: List[OptimalScalingParameter],
    category_lower_bounds: List[OptimalScalingParameter],
    sim: List[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: Dict,
) -> Dict:
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
    if options['method'] == REDUCED:
        last_opt_values = np.asarray([x.value for x in category_upper_bounds])
    elif options['method'] == STANDARD:
        last_opt_values = np.ravel(
            [
                np.asarray([x.value for x in category_lower_bounds]),
                np.asarray([x.value for x in category_upper_bounds]),
            ],
            'F',
        )

    if options['method'] == REDUCED:
        parameter_length = len(category_upper_bounds)
        if len(np.nonzero(last_opt_values)) > 0:
            x0 = last_opt_values
        else:
            x0 = np.linspace(
                np.max([min_all, interval_range]),
                max_all + (interval_range + interval_gap) * parameter_length,
                parameter_length,
            )
    elif options['method'] == STANDARD:
        parameter_length = 2 * len(category_upper_bounds)
        if len(np.nonzero(last_opt_values)) > 0:
            x0 = last_opt_values
        else:
            x0 = np.linspace(0, max_all + interval_range, parameter_length)
    else:
        raise NotImplementedError(
            f"Unknown optimal scaling 'method' {options['method']}. "
            f"Please use {STANDARD} or {REDUCED}."
        )

    if options['reparameterized']:
        x0 = reparameterize_inner_parameters(
            x0, category_upper_bounds, interval_gap, interval_range
        )
        bounds = Bounds(
            [0.0] * parameter_length,
            [max_all + (interval_range + interval_gap) * parameter_length]
            * parameter_length,
        )
        inner_options = {
            'x0': x0,
            'method': 'L-BFGS-B',
            'options': {'maxiter': 2000, 'ftol': 1e-10},
            'bounds': bounds,
        }
    else:
        constraints = get_constraints_for_optimization(
            category_upper_bounds, sim, options
        )

        inner_options = {
            'x0': x0,
            'method': 'SLSQP',
            'options': {'maxiter': 2000, 'ftol': 1e-10, 'disp': None},
            'constraints': constraints,
        }
    return inner_options


def get_min_max(
    inner_parameters: List[OptimalScalingParameter], sim: List[np.ndarray]
) -> Tuple[float, float]:
    """Return minimal and maximal simulation value."""
    sim_all = get_sim_all(inner_parameters, sim)

    min_all = np.min(sim_all)
    max_all = np.max(sim_all)

    return min_all, max_all


def get_sy_all(
    inner_parameters: List[OptimalScalingParameter],
    sy: List[np.ndarray],
    par_idx: int,
):
    """Return model sensitivities for inner parameters and outer parameter index."""
    sy_all = []
    for inner_parameter in inner_parameters:
        for sy_i, mask_i in zip(sy, inner_parameter.ixs):
            sim_sy = sy_i[:, par_idx, :][mask_i]
            # if mask_i.any():
            for sim_sy_i in sim_sy:
                sy_all.append(sim_sy_i)
    return np.array(sy_all)


def get_sim_all(inner_parameters, sim: List[np.ndarray]) -> list:
    """Return model simulations for inner parameters."""
    sim_all = []
    for inner_parameter in inner_parameters:
        for sim_i, mask_i in zip(sim, inner_parameter.ixs):
            sim_x = sim_i[mask_i]
            for sim_x_i in sim_x:
                sim_all.append(sim_x_i)
    return sim_all


def get_surrogate_all(
    inner_parameters: List[OptimalScalingParameter],
    optimal_scaling_bounds: List,
    sim: List[np.ndarray],
    interval_range: float,
    interval_gap: float,
    options: Dict,
):
    """Return surrogate data, lower and upper category bounds."""
    if options['reparameterized']:
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
    inner_parameters: List[OptimalScalingParameter], sim: List[np.ndarray]
) -> float:
    """Calculate weights for objective function."""
    sim_x_all = get_sim_all(inner_parameters, sim)
    eps = 1e-8

    # Three different types of weights
    # v_net = 0
    # for idx in range(len(sim_x_all) - 1):
    #     v_net += np.abs(sim_x_all[idx + 1] - sim_x_all[idx])
    # w_pargett = (0.5 * np.sum(np.abs(sim_x_all)) + v_net + eps)** 2

    # w_squared = np.sum(np.square(sim_x_all)) + eps

    w = np.sum(np.abs(sim_x_all)) + eps

    return w


def compute_interval_constraints(
    inner_parameters: List[OptimalScalingParameter],
    sim: List[np.ndarray],
    options: Dict,
) -> Tuple[float, float]:
    """Compute minimal interval range and gap."""
    # compute constraints on interval size and interval gap size
    # similar to Pargett et al. (2014)
    if 'minGap' not in options:
        eps = 1e-16
    else:
        eps = options['minGap']

    min_simulation, max_simulation = get_min_max(inner_parameters, sim)

    if options['intervalConstraints'] == MAXMIN:
        interval_range = (max_simulation - min_simulation) / (
            2 * len(inner_parameters) + 1
        )
        interval_gap = (max_simulation - min_simulation) / (
            4 * (len(inner_parameters) - 1) + 1
        )
    elif options['intervalConstraints'] == MAX:
        interval_range = max_simulation / (2 * len(inner_parameters) + 1)
        interval_gap = max_simulation / (4 * (len(inner_parameters) - 1) + 1)
    else:
        raise ValueError(
            f"intervalConstraints = "
            f"{options['intervalConstraints']} not implemented. "
            f"Please use {MAX} or {MAXMIN}."
        )
    return interval_range, interval_gap + eps


def reparameterize_inner_parameters(
    original_inner_parameter_values: np.ndarray,
    inner_parameters: List[OptimalScalingParameter],
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
    inner_parameters: List[OptimalScalingParameter],
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
    xs: List[OptimalScalingParameter],
    optimal_scaling_bounds: np.ndarray,
    sim: List[np.ndarray],
    interval_gap: float,
    interval_range: float,
    w: float,
    options: Dict,
) -> float:
    """Compute optimal scaling objective function."""
    obj = 0.0
    if options['reparameterized']:
        optimal_scaling_bounds = undo_inner_parameter_reparameterization(
            optimal_scaling_bounds, xs, interval_gap, interval_range
        )

    for x in xs:
        x_upper, x_lower = get_bounds_for_category(
            x, optimal_scaling_bounds, interval_gap, options
        )
        for sim_i, mask_i in zip(sim, x.ixs):
            # if mask_i.any():
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


def get_bounds_for_category(
    x: OptimalScalingParameter,
    optimal_scaling_bounds: np.ndarray,
    interval_gap: float,
    options: Dict,
) -> Tuple[float, float]:
    """Return upper and lower bound for a specific category x."""
    x_category = int(x.category)

    if options['method'] == REDUCED:
        x_upper = optimal_scaling_bounds[x_category - 1]
        if x_category == 1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        else:
            raise ValueError('Category value needs to be larger than 0.')
    elif options['method'] == STANDARD:
        x_lower = optimal_scaling_bounds[2 * x_category - 2]
        x_upper = optimal_scaling_bounds[2 * x_category - 1]
    else:
        raise NotImplementedError(
            f"Unknown optimal scaling 'method' {options['method']}. "
            f"Please use {REDUCED} or {STANDARD}."
        )
    return x_upper, x_lower


def get_constraints_for_optimization(
    xs: List[OptimalScalingParameter], sim: List[np.ndarray], options: Dict
) -> Dict:
    """Return constraints for inner optimization."""
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(
        xs, sim, options
    )
    if options['method'] == REDUCED:
        a = np.diag(-np.ones(num_categories), -1) + np.diag(
            np.ones(num_categories + 1)
        )
        a = a[:-1, :-1]
        b = np.empty((num_categories,))
        b[0] = interval_range
        b[1:] = interval_range + interval_gap
    elif options['method'] == STANDARD:
        a = np.diag(-np.ones(2 * num_categories), -1) + np.diag(
            np.ones(2 * num_categories + 1)
        )
        a = a[:-1, :]
        a = a[:, :-1]
        b = np.empty((2 * num_categories,))
        b[0] = 0
        b[1::2] = interval_range
        b[2::2] = interval_gap
    ineq_cons = {'type': 'ineq', 'fun': lambda x: a.dot(x) - b}

    return ineq_cons


def save_inner_parameters_to_inner_problem(
    category_upper_bounds: List[OptimalScalingParameter],
    gr,
    problem: OptimalScalingProblem,
    x_inner_opt: Dict,
    sim: List[np.ndarray],
    options: Dict,
):
    """Save inner parameter values to the inner subproblem."""
    interval_range, interval_gap = compute_interval_constraints(
        category_upper_bounds, sim, options
    )

    surrogate_all, x_lower, x_upper = get_surrogate_all(
        category_upper_bounds,
        x_inner_opt['x'],
        sim,
        interval_range,
        interval_gap,
        options,
    )
    problem.groups[gr]['surrogate_data'] = surrogate_all.flatten()

    for inner_parameter in problem.get_cat_ub_parameters_for_group(gr):
        inner_parameter.value = x_upper[inner_parameter.category - 1]
    for inner_parameter in problem.get_cat_lb_parameters_for_group(gr):
        inner_parameter.value = x_lower[inner_parameter.category - 1]
