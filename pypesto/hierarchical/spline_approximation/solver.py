import warnings
from typing import Dict, List

import numpy as np
from scipy.optimize import minimize

from ...C import (
    CURRENT_SIMULATION,
    DATAPOINTS,
    EXPDATA_MASK,
    INNER_NOISE_PARS,
    MAX_DATAPOINT,
    MIN_DATAPOINT,
    MIN_DIFF_FACTOR,
    MIN_SIM_RANGE,
    N_SPLINE_PARS,
    NUM_DATAPOINTS,
    OPTIMIZE_NOISE,
    SCIPY_FUN,
    SCIPY_SUCCESS,
    SCIPY_X,
    InnerParameterType,
)
from ..solver import InnerSolver
from .parameter import SplineInnerParameter
from .problem import SplineInnerProblem

try:
    from amici.parameter_mapping import ParameterMapping
except ImportError:
    pass


class SplineInnerSolver(InnerSolver):
    """Solver of the inner subproblem of spline approximation for nonlinear-monotone data.

    Options
    -------
    min_diff_factor:
        Determines the minimum difference between two consecutive spline
        as ``min_diff_factor * (measurement_range) / n_spline_pars``.
        Default is 1/2.
    """

    def __init__(self, options: Dict = None):
        self.options = {
            **self.get_default_options(),
            **(options or {}),
        }
        self.validate_options()

    def validate_options(self):
        """Validate the current options dictionary."""
        if type(self.options[MIN_DIFF_FACTOR]) is not float:
            raise TypeError(f"{MIN_DIFF_FACTOR} must be of type float.")
        elif self.options[MIN_DIFF_FACTOR] < 0:
            raise ValueError(f"{MIN_DIFF_FACTOR} must be greater than zero.")
        for key in self.options:
            if key not in self.get_default_options():
                raise ValueError(f"Unknown SplineInnerSolver option {key}.")

    def solve(
        self,
        problem: SplineInnerProblem,
        sim: List[np.ndarray],
        amici_sigma: List[np.ndarray],
    ) -> list:
        """Get results for every group (inner optimization problem).

        Parameters
        ----------
        problem:
            InnerProblem from pyPESTO hierarchical.
        sim:
            Simulations from AMICI.
        amici_sigma:
            List of sigmas from AMICI.

        Returns
        -------
        List of optimization results of the inner subproblem.
        """
        inner_results = []
        for group in problem.get_groups_for_xs(InnerParameterType.SPLINE):
            group_dict = problem.groups[group]
            group_dict[CURRENT_SIMULATION] = extract_expdata_using_mask(
                expdata=sim, mask=group_dict[EXPDATA_MASK]
            )
            # Optimize the spline for this group.
            inner_result_for_group = self._optimize_spline(
                inner_parameters=problem.get_free_xs_for_group(group),
                group_dict=group_dict,
            )

            # If the parameters are optimized in the inner problem, we
            # calculate the sigma analytically from the inner result.
            if group_dict[OPTIMIZE_NOISE]:
                group_dict[INNER_NOISE_PARS] = _calculate_sigma_for_group(
                    inner_result=inner_result_for_group,
                    n_datapoints=group_dict[NUM_DATAPOINTS],
                )
            # Otherwise, we extract the sigma from the AMICI noise parameters.
            else:
                group_dict[INNER_NOISE_PARS] = extract_expdata_using_mask(
                    expdata=amici_sigma, mask=group_dict[EXPDATA_MASK]
                )[0]

            # Apply sigma to inner result.
            inner_result_for_group = _calculate_nllh_for_group(
                inner_result=inner_result_for_group,
                sigma=group_dict[INNER_NOISE_PARS],
                n_datapoints=group_dict[NUM_DATAPOINTS],
            )

            inner_results.append(inner_result_for_group)
            save_inner_parameters_to_inner_problem(
                inner_problem=problem,
                s=inner_result_for_group[SCIPY_X],
                group=group,
            )
        return inner_results

    @staticmethod
    def calculate_obj_function(x_inner_opt: list):
        """Calculate the inner objective function value.

        Calculates the inner objective function value from a list of inner
        optimization results returned from `_optimize_spline`.

        Parameters
        ----------
        x_inner_opt:
            List of optimization results

        Returns
        -------
        Inner objective function value.
        """
        if False in (
            x_inner_opt[idx][SCIPY_SUCCESS] for idx in range(len(x_inner_opt))
        ):
            obj = np.inf
            warnings.warn("Inner optimization failed.")
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
        problem: SplineInnerProblem,
        x_inner_opt: List[Dict],
        sim: List[np.ndarray],
        amici_sigma: List[np.ndarray],
        sy: List[np.ndarray],
        amici_ssigma: List[np.ndarray],
        parameter_mapping: ParameterMapping,
        par_opt_ids: List,
        par_sim_ids: List,
        snllh: Dict,
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
        sigma:
            Model noise parameters.
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
            for par_sim, par_opt in condition_map_sim_var.items():
                if (
                    not isinstance(par_opt, str)
                    or par_opt in already_calculated
                ):
                    continue
                # Current fix for scaling/offset parameters in models.
                elif par_sim.startswith('observableParameter'):
                    continue
                # For noise parameters optimized hierarchically, we
                # do not calculate the gradient.
                elif (
                    par_sim.startswith('noiseParameter')
                    and par_opt not in par_opt_ids
                ):
                    continue
                else:
                    already_calculated.add(par_opt)
                par_sim_idx = par_sim_ids.index(par_sim)
                par_opt_idx = par_opt_ids.index(par_opt)
                grad = 0.0

                sy_for_outer_parameter = [
                    sy_cond[:, par_sim_idx, :] for sy_cond in sy
                ]
                ssigma_for_outer_parameter = [
                    ssigma_cond[:, par_sim_idx, :]
                    for ssigma_cond in amici_ssigma
                ]

                for group_idx, group in enumerate(
                    problem.get_groups_for_xs(InnerParameterType.SPLINE)
                ):
                    # Get the reformulated spline parameters
                    s = np.asarray(x_inner_opt[group_idx][SCIPY_X])
                    group_dict = problem.groups[group]

                    measurements = group_dict[DATAPOINTS]
                    sigma = group_dict[INNER_NOISE_PARS]
                    sim_all = group_dict[CURRENT_SIMULATION]
                    N = group_dict[N_SPLINE_PARS]
                    K = group_dict[NUM_DATAPOINTS]

                    sy_all = extract_expdata_using_mask(
                        expdata=sy_for_outer_parameter,
                        mask=group_dict[EXPDATA_MASK],
                    )
                    ssigma_all = extract_expdata_using_mask(
                        expdata=ssigma_for_outer_parameter,
                        mask=group_dict[EXPDATA_MASK],
                    )

                    delta_c, c, n = self._rescale_spline_bases(
                        sim_all=sim_all, N=N, K=K
                    )
                    delta_c_dot, c_dot = calculate_spline_bases_gradient(
                        sim_all=sim_all, sy_all=sy_all, N=N
                    )
                    C = np.diag(-np.ones(N))

                    # For the reformulated problem, mu can be calculated
                    # as the inner gradient at the optimal point s.
                    mu = calculate_inner_gradient_for_obs(
                        s=s,
                        sim_all=sim_all,
                        measurements=measurements,
                        N=N,
                        delta_c=delta_c,
                        c=c,
                        n=n,
                    )
                    min_meas = group_dict[MIN_DATAPOINT]
                    max_meas = group_dict[MAX_DATAPOINT]
                    min_diff = self._get_minimal_difference(
                        measurement_range=max_meas - min_meas,
                        N=N,
                        min_diff_factor=self.options[MIN_DIFF_FACTOR],
                    )

                    # If the spline parameter is at its boundary, the
                    # corresponding Lagrangian multiplier mu is set to 0.
                    min_diff_all = np.full(N, min_diff)
                    min_diff_all[0] = 0.0
                    mu = np.asarray(
                        [
                            mu[i]
                            if np.isclose(s[i] - min_diff_all[i], 0)
                            else 0
                            for i in range(len(s))
                        ]
                    )

                    # Calculate (dJ_ds * ds_dtheta) term only if mu is not all 0
                    ds_grad_term = 0.0
                    if np.any(mu):
                        s_dot = calculate_ds_dtheta(
                            sim_all=sim_all,
                            sy_all=sy_all,
                            measurements=measurements,
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
                        dres_ds = mu
                        ds_grad_term = dres_ds.dot(s_dot)

                    # Let's calculate the (dJ_dy * dy_dtheta) term now:
                    dy_grad_term = calculate_dy_term(
                        sim_all=sim_all,
                        sy_all=sy_all,
                        measurements=measurements,
                        s=s,
                        N=N,
                        delta_c=delta_c,
                        delta_c_dot=delta_c_dot,
                        c=c,
                        c_dot=c_dot,
                        n=n,
                    )

                    # Let's calculate the (dJ_dsigma^2 * dsigma^2_dtheta) term now:
                    if not group_dict[OPTIMIZE_NOISE]:
                        residual_squared = (
                            calculate_objective_function_for_obs(
                                s=s,
                                sim_all=sim_all,
                                measurements=measurements,
                                N=N,
                                delta_c=delta_c,
                                c=c,
                                n=n,
                            )
                        )
                        dJ_dsigma2 = (
                            K / (2 * sigma**2)
                            - residual_squared / sigma**4
                        )
                        dsigma2_dtheta = ssigma_all[0]
                        dsigma_grad_term = dJ_dsigma2 * dsigma2_dtheta
                    # If we optimize the noise hierarchically,
                    # the last term (dJ_dsigma^2 * dsigma^2_dtheta) is always 0
                    # since the sigma is optimized such that dJ_dsigma2=0.
                    else:
                        dsigma_grad_term = 0.0

                    # Combine all terms to get the complete gradient contribution
                    grad += (
                        dy_grad_term / sigma**2
                        + ds_grad_term / sigma**2
                        + dsigma_grad_term
                    )

                snllh[par_opt_idx] = grad

        return snllh

    @staticmethod
    def get_default_options() -> Dict:
        """Return default options for solving the inner problem."""
        options = {
            MIN_DIFF_FACTOR: 1 / 2,
        }
        return options

    def _optimize_spline(
        self,
        inner_parameters: List[SplineInnerParameter],
        group_dict: Dict,
    ):
        """Run optimization for the inner problem.

        Parameters
        ----------
        inner_parameters:
            The spline inner parameters.
        group_dict:
            The group dictionary.
        """
        group_measurements = group_dict[DATAPOINTS]
        current_group_simulation = group_dict[CURRENT_SIMULATION]
        n_datapoints = group_dict[NUM_DATAPOINTS]
        n_spline_pars = group_dict[N_SPLINE_PARS]

        (
            distance_between_bases,
            spline_bases,
            intervals_per_sim,
        ) = self._rescale_spline_bases(
            sim_all=current_group_simulation, N=n_spline_pars, K=n_datapoints
        )

        min_meas = group_dict[MIN_DATAPOINT]
        max_meas = group_dict[MAX_DATAPOINT]
        min_diff = self._get_minimal_difference(
            measurement_range=max_meas - min_meas,
            N=n_spline_pars,
            min_diff_factor=self.options[MIN_DIFF_FACTOR],
        )

        inner_options = self._get_inner_optimization_options(
            inner_parameters=inner_parameters,
            N=n_spline_pars,
            min_meas=min_meas,
            max_meas=max_meas,
            min_diff=min_diff,
        )

        def objective_function_wrapper(x):
            return calculate_objective_function_for_obs(
                s=x,
                sim_all=current_group_simulation,
                measurements=group_measurements,
                N=n_spline_pars,
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
            )

        def inner_gradient_wrapper(x):
            return calculate_inner_gradient_for_obs(
                s=x,
                sim_all=current_group_simulation,
                measurements=group_measurements,
                N=n_spline_pars,
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
            )

        results = minimize(
            objective_function_wrapper,
            jac=inner_gradient_wrapper,
            **inner_options,
        )

        return results

    def _rescale_spline_bases(self, sim_all: np.ndarray, N: int, K: int):
        """Rescale the spline bases.

        Before the optimization of the spline parameters, we have to fix the
        spline bases to some values. We choose to scale them to the current
        simulation. In case of simulations that are very close to each other,
        we choose to scale closely around the average value of the simulations,
        to avoid numerical problems (as we often divide by delta_c).

        Parameters
        ----------
        sim_all:
            The current simulation.
        N:
            The number of spline parameters.
        K:
            The number of simulations.

        Returns
        -------
        distance_between_bases:
            The distance between the spline bases.
        spline_bases:
            The rescaled spline bases.
        intervals_per_sim:
            List of indices of intervals each simulation belongs to.
        """
        min_idx = np.argmin(sim_all)
        max_idx = np.argmax(sim_all)

        min_all = sim_all[min_idx]
        max_all = sim_all[max_idx]

        n = np.ones(K)

        # In case the simulation are very close to each other
        # or even collapse into a single point.
        if max_all - min_all < MIN_SIM_RANGE:
            average_value = (max_all + min_all) / 2
            delta_c = MIN_SIM_RANGE / (N - 1)
            if average_value < (MIN_SIM_RANGE / 2):
                c = np.linspace(0, MIN_SIM_RANGE, N)
            else:
                c = np.linspace(
                    average_value - (MIN_SIM_RANGE / 2),
                    average_value + (MIN_SIM_RANGE / 2),
                    N,
                )
            # Set the n(k) values for the simulations
            for i in range(len(sim_all)):
                n[i] = np.ceil((sim_all[i] - c[0]) / delta_c) + 1
                if n[i] > N:
                    n[i] = N
                    warnings.warn(
                        "Interval for a simulation has been set to a larger value than the number of spline parameters."
                    )
        # In case the simulations are sufficiently apart:
        else:
            delta_c = (max_all - min_all) / (N - 1)
            c = np.linspace(min_all, max_all, N)
            for i in range(len(sim_all)):
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
        measurement_range: float,
        N: int,
        min_diff_factor: float,
    ):
        """Return minimal parameter difference for spline parameters."""
        return min_diff_factor * measurement_range / N

    def _get_inner_optimization_options(
        self,
        inner_parameters: List[SplineInnerParameter],
        N: int,
        min_meas: float,
        max_meas: float,
        min_diff: float,
    ) -> Dict:
        """Return default options for scipy optimizer.

        Returns inner subproblem optimization options including startpoint
        and optimization bounds or constraints, dependent on solver method.

        Parameters
        ----------
        inner_parameters:
            Inner parameters of the spline group.
        N:
            Number of spline parameters.
        min_meas:
            Minimal measurement value.
        max_meas:
            Maximal measurement value.
        min_diff:
            Minimal difference between spline parameters.
        """
        range_all = max_meas - min_meas

        constraint_min_diff = np.full(N, min_diff)
        constraint_min_diff[0] = 0

        last_opt_values = np.asarray([x.value for x in inner_parameters])

        if (last_opt_values > 0).any():
            x0 = last_opt_values
        # In case this is the first inner optimization, initialize the
        # spline parameters to a linear function with a symmetric 60%
        # larger range than the measurement range.
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

        from scipy.optimize import Bounds

        inner_options = {
            "x0": x0,
            "method": "L-BFGS-B",
            "options": {"ftol": 1e-16, "disp": None},
            "bounds": Bounds(lb=constraint_min_diff),
        }

        return inner_options


def _calculate_sigma_for_group(
    inner_result: Dict,
    n_datapoints: int,
):
    """Calculate the noise parameter sigma.

    Parameters
    ----------
    noise_parameters:
        The noise parameters of a group of the inner problem.
    inner_result:
        The inner optimization result.
    """
    sigma = np.sqrt(2 * inner_result[SCIPY_FUN] / (n_datapoints))

    return sigma


def calculate_objective_function_for_obs(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Objective function for reformulated inner spline problem."""
    obj = 0

    for y_k, z_k, n_k in zip(sim_all, measurements, n):
        i = n_k - 1
        sum_s = 0
        sum_s = np.sum(s[:i])
        if i == 0:
            obj += (z_k - s[i]) ** 2
        elif i == N:
            obj += (z_k - sum_s) ** 2
        else:
            obj += (z_k - (y_k - c[i - 1]) * s[i] / delta_c - sum_s) ** 2
    obj = obj / 2
    return obj


def get_spline_mapped_simulations(
    s: np.ndarray,
    sim_all: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Return model simulations mapped using the approximation spline."""
    mapped_simulations = np.zeros(len(sim_all))
    xi = np.zeros(len(s))
    for i in range(len(s)):
        xi[i] = np.sum(s[: i + 1])

    for y_k, n_k, index in zip(sim_all, n, range(len(sim_all))):
        interval_index = n_k - 1
        if interval_index == 0 or interval_index == N:
            mapped_simulations[index] = xi[interval_index]
        else:
            mapped_simulations[index] = (y_k - c[interval_index - 1]) * (
                xi[interval_index] - xi[interval_index - 1]
            ) / delta_c + xi[interval_index - 1]

    return mapped_simulations


def calculate_inner_gradient_for_obs(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Gradient of the objective function for the reformulated inner spline problem."""

    gradient = np.zeros(N)

    for y_k, z_k, n_k in zip(sim_all, measurements, n):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Jacobian array
        sum_s = np.sum(s[:i])
        if i == 0:
            gradient[i] += s[i] - z_k
        elif i == N:
            gradient[:i] += np.full(i, sum_s - z_k)
        else:
            gradient[i] += (
                ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * (y_k - c[i - 1])
                / delta_c
            )
            gradient[:i] += np.full(
                i,
                (y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k,
            )
    return gradient


def calculate_inner_hessian(
    s: np.ndarray,
    sim_all: np.ndarray,
    sigma: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Calculate the hessian of the objective function for the reformulated inner problem."""

    hessian = np.zeros((N, N))

    for y_k, sigma_k, n_k in zip(sim_all, sigma, n):
        sum_s = 0
        i = n_k - 1  # just the iterator to go over the Hessian matrix
        for j in range(i):
            sum_s += s[j]

        hessian[i][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c) ** 2
        for j in range(i):
            hessian[i][j] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            hessian[j][i] += (1 / sigma_k**2) * ((y_k - c[i - 1]) / delta_c)
            for h in range(i):
                hessian[j][h] += 1 / sigma_k**2

    return hessian


def calculate_ds_dtheta(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
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
    """Calculate derivatives of reformulated spline parameters with respect to outer parameter.

    Calculates the derivative of reformulated spline parameters s with respect to the
    dynamical parameter theta. Firstly, we calculate the derivative of the
    first two equations of the necessary optimality conditions of the
    optimization problem with inequality constraints. Then we solve the linear
    system to obtain the derivatives.
    """

    dgrad_dtheta_lhs = np.zeros((N, N))
    dgrad_dtheta_rhs = np.zeros(2 * N)

    for y_k, z_k, y_dot_k, n_k in zip(sim_all, measurements, sy_all, n):
        i = n_k - 1  # just the iterator to go over the matrix
        sum_s = 0
        sum_s = np.sum(s[:i])

        # Calculate dgrad_dtheta in the form of a linear system:
        if i == 0:
            dgrad_dtheta_lhs[i][i] += 1
        elif i == N:
            dgrad_dtheta_lhs = dgrad_dtheta_lhs + np.full((N, N), 1)

        else:
            dgrad_dtheta_lhs[i][i] += (y_k - c[i - 1]) ** 2 / delta_c**2
            dgrad_dtheta_rhs[i] += (
                (2 * (y_k - c[i - 1]) / delta_c * s[i] + sum_s - z_k)
                * (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                / delta_c**2
            )

            dgrad_dtheta_lhs[i, :i] += np.full(i, (y_k - c[i - 1]) / delta_c)
            dgrad_dtheta_lhs[:i, i] += np.full(i, (y_k - c[i - 1]) / delta_c)
            dgrad_dtheta_rhs[:i] += np.full(
                i,
                (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                * s[i]
                / delta_c**2,
            )
            dgrad_dtheta_lhs[:i, :i] += np.full((i, i), 1)

    from scipy import linalg

    constraint_min_diff = np.diag(np.full(N, min_diff))
    constraint_min_diff[0, 0] = 0

    lhs = np.block(
        [
            [dgrad_dtheta_lhs, C],
            [-np.diag(mu), constraint_min_diff - np.diag(s)],
        ]
    )

    ds_dtheta = linalg.lstsq(lhs, dgrad_dtheta_rhs, lapack_driver="gelsy")

    return ds_dtheta[0][:N]


def calculate_dy_term(
    sim_all: np.ndarray,
    sy_all: np.ndarray,
    measurements: np.ndarray,
    s: np.ndarray,
    N: int,
    delta_c: float,
    delta_c_dot: float,
    c: np.ndarray,
    c_dot: np.ndarray,
    n: np.ndarray,
):
    """Calculate the derivative of the objective function for one group with respect to the simulations."""
    df_dy = 0

    for y_k, z_k, y_dot_k, n_k in zip(sim_all, measurements, sy_all, n):
        i = n_k - 1
        sum_s = np.sum(s[:i])
        if i > 0 and i < N:
            df_dy += (
                ((y_k - c[i - 1]) * s[i] / delta_c + sum_s - z_k)
                * s[i]
                * (
                    (y_dot_k - c_dot[i - 1]) * delta_c
                    - (y_k - c[i - 1]) * delta_c_dot
                )
                / delta_c**2
            )
        # There is no i==0 case, because in this case
        # c[0] == y_k and so the derivative is zero.
    return df_dy


def calculate_spline_bases_gradient(
    sim_all: np.ndarray, sy_all: np.ndarray, N: int
):
    """Calculate gradient of the rescaled spline bases."""

    min_idx = np.argmin(sim_all)
    max_idx = np.argmax(sim_all)

    min_all = sim_all[min_idx]
    max_all = sim_all[max_idx]
    # Coming directly from differentiating _rescale_spline_bases
    if sim_all[max_idx] - sim_all[min_idx] < MIN_SIM_RANGE:
        delta_c_dot = 0
        c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
        average_value = (max_all + min_all) / 2
        if average_value < (MIN_SIM_RANGE / 2):
            c_dot = np.full(N, 0)
        else:
            c_dot = np.full(N, (sy_all[max_idx] - sy_all[min_idx]) / 2)
    else:
        delta_c_dot = (sy_all[max_idx] - sy_all[min_idx]) / (N - 1)
        c_dot = np.linspace(sy_all[min_idx], sy_all[max_idx], N)

    return delta_c_dot, c_dot


def extract_expdata_using_mask(
    expdata: List[np.ndarray], mask: List[np.ndarray]
):
    """Extract data from expdata list of arrays for the given mask."""
    return np.concatenate(
        [
            expdata[condition_index][mask[condition_index]]
            for condition_index in range(len(mask))
        ]
    )


def save_inner_parameters_to_inner_problem(
    inner_problem: SplineInnerProblem,
    s: np.ndarray,
    group: int,
) -> None:
    """Save inner parameter values to the inner subproblem.

    Calculates the non-reformulated inner spline parameters from
    the reformulated inner spline parameters and saves them to
    the inner subproblem.

    Parameters
    ----------
    inner_parameters : list
        List of inner parameters.
    s : np.ndarray
        Reformulated inner spline parameters.
    """
    group_dict = inner_problem.groups[group]
    inner_spline_parameters = inner_problem.get_xs_for_group(group)
    inner_noise_parameters = inner_problem.get_noise_parameters_for_group(
        group
    )

    xi = np.zeros(len(s))
    for i in range(len(s)):
        xi[i] = np.sum(s[: i + 1])

    for idx in range(len(inner_spline_parameters)):
        inner_spline_parameters[idx].value = xi[idx]

    sigma = group_dict[INNER_NOISE_PARS]

    if group_dict[OPTIMIZE_NOISE]:
        inner_noise_parameters[0].value = sigma


def _calculate_nllh_for_group(
    inner_result: Dict,
    sigma: float,
    n_datapoints: int,
):
    """Calculate the negative log-likelihood for the group.

    Parameters
    ----------
    inner_result : dict
        Result of the inner problem.
    sigma : float
        Standard deviation of the measurement noise.
    n_datapoints : int
        Number of datapoints.
    """
    inner_result[SCIPY_FUN] = 0.5 * np.log(
        2 * np.pi * sigma**2
    ) * n_datapoints + inner_result[SCIPY_FUN] / (sigma**2)
    return inner_result


def get_monotonicity_measure(measurement, simulation):
    """Get monotonicity measure by calculating inversions.

    Calculate the number of inversions in the simulation data
    with respect to the measurement data.

    Parameters
    ----------
    measurement : np.ndarray
        Measurement data.
    simulation : np.ndarray
        Simulation data.

    Returns
    -------
    inversions : int
        Number of inversions.
    """
    if len(measurement) != len(simulation):
        raise ValueError(
            "Measurement and simulation data must have the same length."
        )

    ordered_simulation = [
        x
        for _, x in sorted(
            zip(measurement, simulation), key=lambda pair: pair[0]
        )
    ]
    ordered_measurement = sorted(simulation)

    inversions = 0
    for i in range(len(ordered_simulation)):
        for j in range(i + 1, len(ordered_simulation)):
            if ordered_simulation[i] > ordered_simulation[j]:
                inversions += 1
            elif (
                ordered_simulation[i] == ordered_simulation[j]
                and ordered_measurement[i] != ordered_measurement[j]
            ):
                inversions += 1

    return inversions
