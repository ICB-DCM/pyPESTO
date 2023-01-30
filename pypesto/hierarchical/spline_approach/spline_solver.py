import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

from ...C import InnerParameterType
from ...optimize import Optimizer
from ..solver import InnerSolver
from .spline_parameter import SplineInnerParameter
from .spline_problem import SplineInnerProblem


class SplineInnerSolver(InnerSolver):
    """
    Solver of the inner subproblem.

    Options
    -------
    spline_ratio:
        Wanted ratio of the number of spline parameters and the number
        of measurements. Can be any positive float. Default value is 1/2.
    inner_optimizer:
        Optimizer of the inner problem. Default is SLSQP.
    minimal_difference:
        If True then the method will constrain minimal spline parameter
        difference. Otherwise there will be no such constrain.
    """

    def __init__(self, optimizer: Optimizer = None, options: Dict = None):

        self.optimizer = optimizer
        self.options = options
        if self.options is None:
            self.options = SplineInnerSolver.get_default_options()
        else:
            if self.options['inner_optimizer'] not in ['SLSQP', 'LS', 'fides']:
                raise ValueError(
                    f"Chosen Inner optimizer {optimizer} is not implemented. Choose from SLSQP, LS or fides"
                )
            if self.options['use_minimal_difference'] not in [True, False]:
                raise ValueError('Minimal difference must be a boolean value.')

    def solve(
        self,
        problem: SplineInnerProblem,
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
    ) -> list:
        """
        Get results for every group (inner optimization problem).

        Parameters
        ----------
        problem:
            InnerProblem from pyPESTO hierarchical.
        sim:
            Simulations from AMICI.
        sigma:
            List of sigmas.
        """
        inner_optimization_results = []

        for group in problem.get_groups_for_xs(InnerParameterType.SPLINE):
            group_dict = problem.groups[group]
            group_dict['noise_parameters'] = extract_expdata_using_mask(
                expdata=sigma, mask=group_dict['expdata_mask']
            )
            group_dict['current_simulation'] = extract_expdata_using_mask(
                expdata=sim, mask=group_dict['expdata_mask']
            )

            inner_optimization_results_per_group = self._optimize_spline(
                inner_parameters=problem.get_free_xs_for_group(group),
                group_dict=group_dict,
            )
            inner_optimization_results.append(
                inner_optimization_results_per_group
            )
            save_inner_parameters_to_inner_problem(
                inner_parameters=problem.get_xs_for_group(group),
                s=inner_optimization_results_per_group['x'],
            )
        return inner_optimization_results

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """
        Calculate the inner objective function from a list of inner
        optimization results returned from solve.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results
        """
        if False in [
            x_inner_opt[idx]['success'] for idx in range(len(x_inner_opt))
        ]:
            obj = np.inf
            warnings.warn(f"Inner optimization failed.")
        else:
            obj = np.sum(
                [x_inner_opt[idx]['fun'] for idx in range(len(x_inner_opt))]
            )
        return obj

    @staticmethod
    def ls_calculate_obj_function(x_inner_opt: list):
        """
        Calculate the inner objective function from a list of inner
        optimization results returned from solve
        using the least squares inner optimizer.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results
        """
        obj = np.sum(
            [x_inner_opt[idx]["fun"][0] for idx in range(len(x_inner_opt))]
        )
        return obj

    def calculate_gradients(
        self,
        problem: SplineInnerProblem,
        x_inner_opt,
        sim,
        sy,
        parameter_mapping,
        par_opt_ids,
        snllh,
        sigma,
    ):
        """
        Calculate the gradient of the objective function with respect to
        the outer dynaical parameters.

        Parameters
        ----------
        problem:
            SplineProblem
        """
        condition_map_sim_var = parameter_mapping[0].map_sim_var
        # TODO: Doesn't work with condition specific parameters
        for par_sim, par_opt in condition_map_sim_var.items():
            if not isinstance(par_opt, str):
                continue
            # check this again TODO
            par_opt_idx = par_opt_ids.index(par_opt)
            grad = 0.0
            sy_for_outer_parameter = [
                sy_cond[:, par_opt_idx, :] for sy_cond in sy
            ]

            for group_idx, group in enumerate(
                problem.get_groups_for_xs(InnerParameterType.SPLINE)
            ):
                # the reformulated spline parameters
                s = np.asarray(x_inner_opt[group_idx]["x"])
                group_dict = problem.groups[group]

                measurements = group_dict['datapoints']
                sigma = group_dict['noise_parameters']
                sim_all = group_dict['current_simulation']
                N = group_dict['n_spline_pars']
                K = group_dict['n_datapoints']

                sy_all = extract_expdata_using_mask(
                    expdata=sy_for_outer_parameter,
                    mask=group_dict['expdata_mask'],
                )

                delta_c, c, n = self._rescale_spline_bases(
                    sim_all=sim_all, N=N, K=K
                )
                delta_c_dot, c_dot = get_spline_bases_gradient(
                    sim_all=sim_all, sy_all=sy_all, N=N
                )

                C = np.diag(-np.ones(N))
                # For the reformulated problem, mu can be calculated
                # as the inner gradient at the optimal point s.
                mu = get_inner_gradient(
                    s=s,
                    sim_all=sim_all,
                    measurements=measurements,
                    sigma=sigma,
                    N=N,
                    delta_c=delta_c,
                    c=c,
                    n=n,
                )
                min_meas = group_dict['min_datapoint']
                max_meas = group_dict['max_datapoint']
                min_diff = self._get_minimal_difference(
                    min_meas=min_meas,
                    max_meas=max_meas,
                    N=N,
                    use_minimal_difference=self.options[
                        'use_minimal_difference'
                    ],
                )

                # Correcting for small errors in optimization/calculations
                # TODO should I do this?
                for i in range(len(mu)):
                    if abs(mu[i]) < 1e-5:
                        mu[i] = 0
                # Calculate df_ds term only if mu is not all 0
                if np.any(mu):
                    s_dot = get_ds_dtheta(
                        sim_all=sim_all,
                        sy_all=sy_all,
                        measurements=measurements,
                        sigma=sigma,
                        s=s,
                        C=C,
                        mu=mu,
                        N=N,
                        delta_c=delta_c,
                        delta_c_dot=delta_c_dot,
                        c=c,
                        c_dot=c_dot,
                        n=n,
                        min_diff=min_diff,
                    )
                    df_ds = mu
                    grad += df_ds.dot(s_dot)

                # Let's calculate the df_dyk term now:
                df_dyk = get_df_dyk(
                    sim_all=sim_all,
                    sy_all=sy_all,
                    measurements=measurements,
                    sigma=sigma,
                    s=s,
                    N=N,
                    delta_c=delta_c,
                    delta_c_dot=delta_c_dot,
                    c=c,
                    c_dot=c_dot,
                    n=n,
                )

                grad += df_dyk
            snllh[par_opt_idx] = grad

        return snllh

    @staticmethod
    def get_default_options() -> Dict:
        """
        Return default options for solving the inner problem,
        if no options provided
        """
        options = {
            "inner_optimizer": 'SLSQP',
            "use_minimal_difference": True,
        }
        return options

    def _optimize_spline(
        self,
        inner_parameters: List[SplineInnerParameter],
        group_dict: Dict,
    ):
        """Run optimization for inner problem"""

        group_measurements = group_dict['datapoints']
        group_noise_parameters = group_dict['noise_parameters']
        current_group_simulation = group_dict['current_simulation']
        n_datapoints = group_dict['n_datapoints']
        n_spline_pars = group_dict['n_spline_pars']

        (
            distance_between_bases,
            spline_bases,
            intervals_per_sim,
        ) = self._rescale_spline_bases(
            sim_all=current_group_simulation, N=n_spline_pars, K=n_datapoints
        )

        min_meas = group_dict['min_datapoint']
        max_meas = group_dict['max_datapoint']
        min_diff = self._get_minimal_difference(
            min_meas=min_meas,
            max_meas=max_meas,
            N=n_spline_pars,
            use_minimal_difference=self.options['use_minimal_difference'],
        )

        inner_options = self._get_inner_optimization_options(
            inner_parameters=inner_parameters,
            N=n_spline_pars,
            min_meas=min_meas,
            max_meas=max_meas,
            min_diff=min_diff,
        )

        def objective_function(x):
            return get_objective_function(
                s=x,
                sim_all=current_group_simulation,
                measurements=group_measurements,
                sigma=group_noise_parameters,
                N=n_spline_pars,
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
            )

        def inner_gradient(x):
            return get_inner_gradient(
                s=x,
                sim_all=current_group_simulation,
                measurements=group_measurements,
                sigma=group_noise_parameters,
                N=n_spline_pars,
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
            )

        if self.options['inner_optimizer'] == 'SLSQP':
            results = minimize(
                objective_function, jac=inner_gradient, **inner_options
            )
            results["x"][0] = results["x"][0].clip(min=0)
            results["x"][1:] = results["x"][1:].clip(min=min_diff)

        elif self.options['inner_optimizer'] == 'LS':
            results = least_squares(
                objective_function,
                inner_options['x0'],
                jac=inner_gradient,
                bounds=(0, np.inf),
            )

        elif self.options['inner_optimizer'] == 'fides':
            import fides

            def inner_hessian(x):
                return get_inner_hessian(
                    s=x,
                    sim_all=current_group_simulation,
                    measurements=group_measurements,
                    sigma=group_noise_parameters,
                    N=n_spline_pars,
                    delta_c=distance_between_bases,
                    c=spline_bases,
                    n=intervals_per_sim,
                )

            def fides_objective_function(x):
                return (
                    get_objective_function(
                        s=x,
                        sim_all=current_group_simulation,
                        measurements=group_measurements,
                        sigma=group_noise_parameters,
                        N=n_spline_pars,
                        delta_c=distance_between_bases,
                        c=spline_bases,
                        n=intervals_per_sim,
                    ),
                    get_inner_gradient(
                        s=x,
                        sim_all=current_group_simulation,
                        measurements=group_measurements,
                        sigma=group_noise_parameters,
                        N=n_spline_pars,
                        delta_c=distance_between_bases,
                        c=spline_bases,
                        n=intervals_per_sim,
                    ),
                    get_inner_hessian(
                        s=x,
                        sim_all=current_group_simulation,
                        measurements=group_measurements,
                        sigma=group_noise_parameters,
                        N=n_spline_pars,
                        delta_c=distance_between_bases,
                        c=spline_bases,
                        n=intervals_per_sim,
                    ),
                )

            lower_bounds = np.full(n_spline_pars, min_diff)
            upper_bounds = np.full(n_spline_pars, np.inf)
            lower_bounds[0] = 0
            opt_fides = fides.Optimizer(
                fides_objective_function, ub=upper_bounds, lb=lower_bounds
            )

            results = opt_fides.minimize(inner_options['x0'])
        # TODO should I be clipping like this?
        results["x"][0] = results["x"][0].clip(min=0)
        results["x"][1:] = results["x"][1:].clip(min=min_diff)

        return results

    def _rescale_spline_bases(self, sim_all: np.ndarray, N: int, K: int):
        """Rescale the spline bases"""

        min_all = sim_all[0]
        max_all = sim_all[0]
        min_idx = 0
        max_idx = 0

        for idx in range(len(sim_all)):
            if sim_all[idx] > max_all:
                max_all = sim_all[idx]
                max_idx = idx
            if sim_all[idx] < min_all:
                min_all = sim_all[idx]
                min_idx = idx

        n = np.ones(K)

        # In case the simulation are very close to each other
        # or even collapse into a single point (e.g. steady-state) (TODO add hyperparameter for 1e-6?)
        if max_all - min_all < 1e-6:
            average_value = (max_all + min_all) / 2
            if average_value < 5e-7:
                delta_c = 1e-6 / (N - 1)
                c = np.linspace(0, 1e-6, N)
            else:
                delta_c = 1e-6 / (N - 1)
                c = np.linspace(average_value - 5e-7, average_value + 5e-7, N)
            # Set the n(k) values for the simulations
            for i in range(len(sim_all)):
                n[i] = np.ceil((sim_all[i] - c[0]) / delta_c) + 1
                if n[i] > N:
                    n[i] = N
                    warnings.warn(
                        f"Interval for a sim has been set to a larger value than the number of spline parameters."
                    )
        # In case the simulations are sufficiently apart:
        else:
            delta_c = (max_all - min_all) / (N - 1)
            c = np.linspace(min_all, max_all, N)
            for i in range(len(sim_all)):
                # FIXME what if there are multiple maximal or minimal incides
                if i == max_idx:
                    n[i] = N
                elif i == min_idx:
                    n[i] = 1
                else:
                    n[i] = np.ceil((sim_all[i] - c[0]) / delta_c) + 1
                if n[i] > N:
                    n[i] = N

        n = n.astype(int)
        return delta_c, c, n

    def _get_minimal_difference(
        self,
        min_meas: float,
        max_meas: float,
        N: int,
        use_minimal_difference: bool,
    ):
        """
        Returns minimal parameter difference for spline parameters.
        """
        if use_minimal_difference:
            min_diff = (max_meas - min_meas) / (2 * N)
        else:
            min_diff = 0
        return min_diff

    def _get_inner_optimization_options(
        self,
        inner_parameters: List[SplineInnerParameter],
        N: int,
        min_meas: float,
        max_meas: float,
        min_diff: float,
    ) -> Dict:

        """Return options for optimization"""
        range_all = max_meas - min_meas

        constraint_min_diff = np.full(N, min_diff)
        constraint_min_diff[0] = 0

        last_opt_values = np.asarray([x.value for x in inner_parameters])

        if len(np.nonzero(last_opt_values)) > 0:
            x0 = last_opt_values
        else:
            x0 = np.full(
                N,
                (
                    max_meas
                    + 0.3 * range_all
                    - np.max([min_meas - 0.3 * range_all, 0])
                )
                / (N - 1),
            )
            x0[0] = np.max([min_meas - 0.3 * range_all, 0])

        inner_options = {
            "x0": x0,
            "method": "SLSQP",
            "options": {"maxiter": 2000, "ftol": 1e-10, "disp": None},
            "constraints": {
                "type": "ineq",
                "fun": lambda x: x - constraint_min_diff,
            },
        }

        return inner_options


def get_objective_function(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Objective function for problem reformulation
    with s as parmaeter splines."""
    obj = 0

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):
        i = n_k - 1
        sum_s = 0
        for j in range(i):
            sum_s += s[j]
        if i == 0:
            obj += (1 / sigma_k**2) * (z_k - s[i]) ** 2
        elif i == N:
            obj += (1 / sigma_k**2) * (z_k - sum_s) ** 2
        else:
            obj += (1 / sigma_k**2) * (
                z_k - (y_k - c[i - 1]) * s[i] / delta_c - sum_s
            ) ** 2
    obj = obj / 2
    return obj


def get_inner_gradient(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Gradient of the objective function with respect
    to the spline parameters for the reformulated inner problem"""

    Gradient = np.zeros(N)

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):

        weight_k = 1 / sigma_k**2
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Jacobian array
        for j in range(i):
            sum_s += s[j]
        if i == 0:
            Gradient[i] += weight_k * (s[i] - z_k)
        elif i == N:
            for j in range(i):
                Gradient[j] += weight_k * (sum_s - z_k)
        else:
            Gradient[i] += (
                weight_k
                * ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * (y_k - c[i - 1])
                / delta_c
            )
            Gradient[:i] += np.full(
                i,
                weight_k * ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k),
            )
    return Gradient


def get_inner_hessian(
    optimal_s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Hessian of the objective function with respect
    to the spline parameters for the reformulated inner problem"""

    Hessian = np.zeros((N, N))

    for y_k, z_k, sigma_k, n_k in zip(sim_all, measurements, sigma, n):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Hessian matrix
        for j in range(i):
            sum_s += optimal_s[j]
        # ALSO MAKE THE LAST MONOTONICITY STEP
        Hessian[i][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c) ** 2
        for j in range(i):
            Hessian[i][j] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            Hessian[j][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            for h in range(i):
                Hessian[j][h] += 1 / sigma_k**2
    # print(np.linalg.eig(Hessian))
    return Hessian


def get_ds_dtheta(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    s: np.ndarray,
    C: np.ndarray,
    mu: np.ndarray,
    N: int,
    delta_c: float,
    delta_c_dot: float,
    c: np.ndarray,
    c_dot: np.ndarray,
    n: np.ndarray,
    min_diff: float,
):
    """Calculates the derivative of reformulated spline parameters s with respect to the
    dynamical parameter theta. Look at 'get_dxi_dtheta()' for details."""

    Jacobian_derivative = np.zeros((N, N))
    rhs = np.zeros(2 * N)

    for y_k, z_k, y_dot_k, sigma_k, n_k in zip(
        sim_all, measurements, sy_all, sigma, n
    ):

        i = n_k - 1  # just the iterator to go over the Jacobian matrix
        weight_k = 1 / sigma_k**2
        sum_s = 0
        for j in range(i):
            sum_s += s[j]

        # calculate the Jacobian derivative:
        if i == 0:
            Jacobian_derivative[i][i] += weight_k
        elif i == N:
            Jacobian_derivative = Jacobian_derivative + np.full(
                (N, N), weight_k
            )

        else:
            Jacobian_derivative[i][i] += (
                weight_k * (y_k - c[i - 1]) ** 2 / delta_c**2
            )
            rhs[i] += (
                weight_k
                * (2 * (y_k - c[i - 1]) / delta_c * s[i] + sum_s - z_k)
                * (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                / delta_c**2
            )
            if i > 0:
                Jacobian_derivative[i, :i] += np.full(
                    i, weight_k * (y_k - c[i - 1]) / delta_c
                )
                Jacobian_derivative[:i, i] += np.full(
                    i, weight_k * (y_k - c[i - 1]) / delta_c
                )
                rhs[:i] += np.full(
                    i,
                    weight_k
                    * (
                        (y_dot_k - c_dot[i - 1]) * delta_c
                        - (y_k - c[i - 1]) * delta_c_dot
                    )
                    * s[i]
                    / delta_c**2,
                )
                Jacobian_derivative[:i, :i] += np.full((i, i), weight_k)

    from scipy import linalg

    constraint_min_diff = np.diag(np.full(N, min_diff))
    constraint_min_diff[0][0] = 0
    lhs = np.block(
        [
            [Jacobian_derivative, C],
            [-np.diag(mu), constraint_min_diff - np.diag(s)],
        ]
    )
    ds_dtheta = linalg.lstsq(lhs, rhs, lapack_driver="gelsy")

    return ds_dtheta[0][:N]


def get_df_dyk(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    s: np.ndarray,
    N: int,
    delta_c: float,
    delta_c_dot: float,
    c: np.ndarray,
    c_dot: np.ndarray,
    n: np.ndarray,
):
    df_dyk = 0

    for y_k, z_k, y_dot_k, sigma_k, n_k in zip(
        sim_all, measurements, sy_all, sigma, n
    ):

        i = n_k - 1
        sum_s = 0
        for j in range(i):
            sum_s += s[j]
        if i > 0 and i < N:
            df_dyk += (
                (1 / sigma_k**2)
                * ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * s[i]
                * (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                / delta_c**2
            )
    return df_dyk


def get_spline_bases_gradient(sim_all: np.ndarray, sy_all: np.ndarray, N: int):
    """Get gradient of the rescaled spline bases"""

    min_idx = 0
    max_idx = 0
    # FIXME WE STILL have a problem here, what if there are multiple
    # simulations with same value y_max or y_min?... Which one?
    for idx in range(len(sim_all)):
        if sim_all[idx] > sim_all[max_idx]:
            max_idx = idx
        if sim_all[idx] < sim_all[min_idx]:
            min_idx = idx

    if sim_all[max_idx] - sim_all[min_idx] < 1e-6:
        delta_c_dot = 0
        c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
    else:
        delta_c_dot = (sy_all[max_idx] - sy_all[min_idx]) / (N - 1)
        c_dot = np.linspace(sy_all[min_idx], sy_all[max_idx], N)

    return delta_c_dot, c_dot


def extract_expdata_using_mask(
    expdata: List[np.ndarray], mask: List[np.ndarray]
):
    """Extract data from expdata list of arrays for the
    given mask."""
    return np.concatenate(
        [
            expdata[condition_index][mask[condition_index]]
            for condition_index in range(len(mask))
        ]
    )


def save_inner_parameters_to_inner_problem(
    inner_parameters,
    s,
):
    xi = np.zeros(len(s))
    for i in range(len(s)):
        for j in range(i, len(s)):
            xi[j] += s[i]

    for idx in range(len(inner_parameters)):
        inner_parameters[idx].value = xi[idx]


def get_monotonicity_measure(quantitative_data, sim_all):
    """Get monotonicity measure by calculating inversions"""
    DeprecationWarning("Outdated function.")
    return
    quantitative_data['simulation'] = sim_all
    quantitative_data = quantitative_data.sort_values(by=['measurement'])
    inversions = 0
    ordered_simulation = quantitative_data.simulation.values
    measurement = quantitative_data.measurement.values
    for i in range(len(ordered_simulation)):
        for j in range(i + 1, len(ordered_simulation)):
            if ordered_simulation[i] > ordered_simulation[j]:
                inversions += 1
            elif (
                ordered_simulation[i] == ordered_simulation[j]
                and measurement[i] != measurement[j]
            ):
                inversions += 1

    return inversions


def get_gradient_old(
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Gradient of the objective function with respect
    to the spline parameters for the non-reformulated problem.
    Returns the linearized form of the gradient:

    gradient = lhs_matrix * xi - rhs

    """
    DeprecationWarning(
        "Deprecated function due to reformulation of the problem."
    )
    return
    lhs_matrix = np.zeros((N, N))
    rhs = np.zeros(N)

    for y_k, z_k, sigma_k in zip(sim_all, measurements, sigma):
        n = math.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1  # just the iterator to go over the Jacobian matrix
        if n < N + 1:
            if n > 1:
                lhs_matrix[i][i - 1] += (
                    (1 / sigma_k**2)
                    * (y_k - c[0] - (n - 2) * delta_c)
                    * (c[0] + (n - 1) * delta_c - y_k)
                )
            lhs_matrix[i][i] += (1 / sigma_k**2) * (
                y_k - c[0] - (n - 2) * delta_c
            ) ** 2
            rhs[i] += (
                (1 / sigma_k**2)
                * z_k
                * (y_k - c[0] - (n - 2) * delta_c)
                * delta_c
            )
        if n > 1 and n < N + 1:
            lhs_matrix[i - 1][i - 1] += (1 / sigma_k**2) * (
                c[0] + (n - 1) * delta_c - y_k
            ) ** 2
            if n < N + 1:
                lhs_matrix[i - 1][i] += (
                    (1 / sigma_k**2)
                    * (y_k - c[0] - (n - 2) * delta_c)
                    * (c[0] + (n - 1) * delta_c - y_k)
                )
            rhs[i - 1] += (
                (1 / sigma_k**2)
                * z_k
                * (c[0] + (n - 1) * delta_c - y_k)
                * delta_c
            )
        if n == N + 1:
            lhs_matrix[i - 1][i - 1] += (1 / sigma_k**2) * delta_c**2
            rhs[i - 1] += (1 / sigma_k**2) * z_k * delta_c**2
    lhs_matrix = np.divide(lhs_matrix, delta_c**2)
    rhs = np.divide(rhs, delta_c**2)
    return lhs_matrix, rhs


def obj_spline_old(
    xi: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Objective function for non-reformulated problem
    with xi as spline parameters"""

    DeprecationWarning(
        "Deprecated function due to reformulation of the problem."
    )
    return
    obj = 0
    for y_k, z_k, sigma_k in zip(sim_all, measurements, sigma):
        n = np.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1
        if n < N + 1 and n > 1:
            obj += (1 / sigma_k**2) * (
                z_k
                - (y_k - c[0] - delta_c * (n - 2))
                * (xi[i] - xi[i - 1])
                / delta_c
                - xi[i - 1]
            ) ** 2
        elif n == N + 1:
            obj += (1 / sigma_k**2) * (z_k - xi[i - 1]) ** 2
        elif n == 1:
            obj += (1 / sigma_k**2) * (z_k - y_k * xi[i] / c[0]) ** 2
        obj += np.log(2 * np.pi * sigma_k**2)
    obj = obj / 2
    return obj


def get_mu_old(Jacobian, rhs, xi, C):
    from scipy import linalg
    from scipy.optimize import least_squares, minimize

    """
    Calculate the Langrange multipliers mu for the non-reformulated problem.
    For the reformulated problem, they equal the gradient.
    """
    DeprecationWarning(
        "Deprecated function due to reformulation of the problem."
    )
    return

    C_transpose = C.transpose()

    rhs_mu = -(Jacobian.dot(xi) - rhs)

    def mu_obj(x):
        return (C_transpose.dot(x) - rhs_mu).sum()

    constraints = [
        {"type": "ineq", "fun": lambda x: x},
        {
            "type": "ineq",
            "fun": lambda x: np.full(len(xi), 1e-6)
            - np.multiply(x, C.dot(xi)),
        },
    ]

    inner_options = {
        "x0": np.zeros(len(xi)),
        "method": "SLSQP",
        "options": {"maxiter": 2000, "ftol": 1e-10, "disp": True},
        "constraints": constraints,
    }

    mu_slsqp = minimize(mu_obj, **inner_options)
    print("mu_slsqp: ", mu_slsqp["x"])
    print("residue: ", (C_transpose.dot(mu_slsqp["x"]) - rhs_mu).sum())

    return mu_slsqp["x"]


def get_dxi_dtheta_old(
    gr,
    sim_all,
    sy_all,
    measurements,
    xi,
    C,
    mu,
    sigma,
    N,
    delta_c,
    delta_c_dot,
    c,
    c_dot,
    n,
):
    """
    Calculates the derivative of spline parameters xi with respect to the
    dynamical parameter theta.
    Firstly, we calculate the derivative of the first two equations of
    the necessary optimality conditions of the optimization problem with
    inequality constraints. Then we solve the linear system to obtain the
    derivatives.
    The derivative of the gradient with respect to the dynamical parameter
    theta is calculated in the linearlized form with respect to dxi_dtheta:

    d_{theta} (gradient_{xi}) =
     = gradient_derivative_lhs * dxi_dtheta - gradient_derivative_rhs

    In the thesis, we noted gradient_derivative_lhs and gradient_derivative_rhs
    as M_i and R_i, respectively.
    """
    DeprecationWarning(
        "Deprecated function due to reformulation of the problem."
    )
    return

    gradient_derivative_lhs = np.zeros((N, N))
    gradient_derivative_rhs = np.zeros(2 * N)

    for y_k, z_k, y_dot_k, sigma_k in zip(
        sim_all, measurements, sy_all, sigma
    ):
        n = np.ceil((y_k - c[0]) / delta_c) + 1
        if n == 0:
            n = 1
        if n > N:
            n = N + 1
        i = n - 1  # just the iterator to go over the Jacobian matrix
        # calculate the Jacobian derivative:
        if n == N + 1:
            # print("U SEDAM SAAAAM LEEEEL")
            gradient_derivative_lhs[i - 1][i - 1] += (
                1 / sigma_k**2
            ) * delta_c**2
            # rhs[i-1] += -2*w_dot*( xi[i-1] - z_k)* delta_c**2 NEED TO ADD HERE IF SIGMA IS OPTIMIZED
        else:
            if n < N + 1:
                if n > 1:
                    gradient_derivative_lhs[i][i - 1] += (
                        (1 / sigma_k**2)
                        * (y_k - c[0] - (n - 2) * delta_c)
                        * (c[0] + (n - 1) * delta_c - y_k)
                    )
                gradient_derivative_lhs[i][i] += (1 / sigma_k**2) * (
                    y_k - c[0] - (n - 2) * delta_c
                ) ** 2
                if n > 1:
                    gradient_derivative_rhs[i] += -(1 / sigma_k**2) * xi[
                        i - 1
                    ] * y_dot_k * (
                        2 * c[0] + (2 * n - 3) * delta_c - 2 * y_k
                    ) - (
                        1 / sigma_k**2
                    ) * xi[
                        i - 1
                    ] * (
                        y_k - c[0] - (n - 2) * delta_c
                    ) * (
                        c[0] + (n - 1) * delta_c - y_k
                    )
                gradient_derivative_rhs[i] += (
                    -(1 / sigma_k**2)
                    * xi[i]
                    * 2
                    * y_dot_k
                    * (y_k - c[0] - (n - 2) * delta_c)
                )
                gradient_derivative_rhs[i] += (
                    (1 / sigma_k**2) * z_k * delta_c * y_dot_k
                )
            if n > 1:
                gradient_derivative_lhs[i - 1][i - 1] += (1 / sigma_k**2) * (
                    c[0] + (n - 1) * delta_c - y_k
                ) ** 2
                if n < N + 1:
                    gradient_derivative_lhs[i - 1][i] += (
                        (1 / sigma_k**2)
                        * (y_k - c[0] - (n - 2) * delta_c)
                        * (c[0] + (n - 1) * delta_c - y_k)
                    )
                if n < N + 1:
                    gradient_derivative_rhs[i - 1] += (
                        -(1 / sigma_k**2)
                        * xi[i]
                        * y_dot_k
                        * (2 * c[0] + (2 * n - 3) * delta_c - 2 * y_k)
                    )
                gradient_derivative_rhs[i - 1] += (
                    -(1 / sigma_k**2)
                    * xi[i - 1]
                    * 2
                    * y_dot_k
                    * (y_k - c[0] - (n - 1) * delta_c)
                )
                gradient_derivative_rhs[i - 1] += (
                    -(1 / sigma_k**2) * z_k * delta_c * y_dot_k
                )

    gradient_derivative_lhs = np.divide(
        gradient_derivative_lhs, ((delta_c) ** 2)
    )
    gradient_derivative_rhs = np.divide(
        gradient_derivative_rhs, ((delta_c) ** 2)
    )

    if np.all((mu == 0)):
        from scipy import linalg

        gradient_derivative_rhs = gradient_derivative_rhs[:N]
        lhs = gradient_derivative_lhs

        dxi_dtheta = linalg.lstsq(lhs, gradient_derivative_rhs)
        return dxi_dtheta[0]
    else:
        from scipy.sparse import csc_matrix, linalg

        lhs = np.block(
            [
                [gradient_derivative_lhs, C.transpose()],
                [(mu * C.transpose()).transpose(), np.diag(C.dot(xi))],
            ]
        )

        lhs_sp = csc_matrix(lhs)

        dxi_dtheta = linalg.spsolve(lhs_sp, gradient_derivative_rhs)
        return dxi_dtheta[:N]
