from __future__ import annotations

import warnings

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
    REGULARIZATION_FACTOR,
    REGULARIZE_SPLINE,
    SCIPY_FUN,
    SCIPY_SUCCESS,
    SCIPY_X,
    InnerParameterType,
)
from ..base_solver import InnerSolver
from .parameter import SplineInnerParameter
from .problem import SemiquantProblem

try:
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass


class SemiquantInnerSolver(InnerSolver):
    """Solver of the inner subproblem of spline approximation for nonlinear-monotone data.

    Options
    -------
    min_diff_factor:
        Determines the minimum difference between two consecutive spline
        as ``min_diff_factor * (measurement_range) / n_spline_pars``.
        Default is 1/2.
    """

    def __init__(self, options: dict = None):
        self.options = {
            **self.get_default_options(),
            **(options or {}),
        }
        self.validate_options()

    def validate_options(self):
        """Validate the current options dictionary."""
        if not isinstance(self.options[MIN_DIFF_FACTOR], float):
            raise TypeError(f"{MIN_DIFF_FACTOR} must be of type float.")
        elif self.options[MIN_DIFF_FACTOR] < 0:
            raise ValueError(f"{MIN_DIFF_FACTOR} must not be negative.")

        elif not isinstance(self.options[REGULARIZE_SPLINE], bool):
            raise TypeError(f"{REGULARIZE_SPLINE} must be of type bool.")
        if self.options[REGULARIZE_SPLINE]:
            if not isinstance(self.options[REGULARIZATION_FACTOR], float):
                raise TypeError(
                    f"{REGULARIZATION_FACTOR} must be of type float."
                )
            elif self.options[REGULARIZATION_FACTOR] < 0:
                raise ValueError(
                    f"{REGULARIZATION_FACTOR} must not be negative."
                )

        for key in self.options:
            if key not in self.get_default_options():
                raise ValueError(f"Unknown SplineInnerSolver option {key}.")

    def solve(
        self,
        problem: SemiquantProblem,
        sim: list[np.ndarray],
        amici_sigma: list[np.ndarray],
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

            # If the noise parameters are optimized in the outer problem,
            # extract them from amici return data.
            if not group_dict[OPTIMIZE_NOISE]:
                group_dict[INNER_NOISE_PARS] = extract_expdata_using_mask(
                    expdata=amici_sigma, mask=group_dict[EXPDATA_MASK]
                )[0]

            # Optimize the spline for this group.
            inner_result_for_group = self._optimize_spline(
                inner_parameters=problem.get_free_xs_for_group(group),
                group_dict=group_dict,
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
            warnings.warn(
                "Inner optimization failed.",
                stacklevel=2,
            )
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
        problem: SemiquantProblem,
        x_inner_opt: list[dict],
        sim: list[np.ndarray],
        amici_sigma: list[np.ndarray],
        sy: list[np.ndarray],
        amici_ssigma: list[np.ndarray],
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
                par_sim_idx = par_sim_ids.index(par_sim)
                par_opt_idx = par_opt_ids.index(par_opt)
                grad = 0.0

                sy_for_outer_parameter = [
                    (
                        sy_cond[:, par_edata_indices.index(par_sim_idx), :]
                        if par_sim_idx in par_edata_indices
                        else np.zeros(sy_cond[:, 0, :].shape)
                    )
                    for sy_cond, par_edata_indices in zip(
                        sy, par_edatas_indices
                    )
                ]
                ssigma_for_outer_parameter = [
                    (
                        ssigma_cond[:, par_edata_indices.index(par_sim_idx), :]
                        if par_sim_idx in par_edata_indices
                        else np.zeros(ssigma_cond[:, 0, :].shape)
                    )
                    for ssigma_cond, par_edata_indices in zip(
                        amici_ssigma, par_edatas_indices
                    )
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

                    # For the reformulated problem, mu can be calculated
                    # as the inner gradient at the optimal point s.
                    mu = _calculate_nllh_gradient_for_group(
                        s=s,
                        sim_all=sim_all,
                        measurements=measurements,
                        N=N,
                        delta_c=delta_c,
                        c=c,
                        n=n,
                        regularization_factor=self.options[
                            REGULARIZATION_FACTOR
                        ],
                        regularize_spline=self.options[REGULARIZE_SPLINE],
                        group_dict=group_dict,
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
                            (
                                mu[i]
                                if np.isclose(s[i] - min_diff_all[i], 0)
                                else 0
                            )
                            for i in range(len(s))
                        ]
                    )

                    # Calculate the (dJ_dy * dy_dtheta) term:
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

                    # Calculate the (dJ_dsigma^2 * dsigma^2_dtheta) term:
                    if not group_dict[OPTIMIZE_NOISE]:
                        residual_squared = _calculate_residuals_for_group(
                            s=s,
                            sim_all=sim_all,
                            measurements=measurements,
                            N=N,
                            delta_c=delta_c,
                            c=c,
                            n=n,
                        )
                        dJ_dsigma2 = (
                            K / (2 * sigma**2) - residual_squared / sigma**4
                        )
                        dsigma2_dtheta = ssigma_all[0] * sigma
                        dsigma_grad_term = dJ_dsigma2 * dsigma2_dtheta
                    # If we optimize the noise hierarchically,
                    # the last term (dJ_dsigma^2 * dsigma^2_dtheta) is always 0
                    # since the sigma is optimized such that dJ_dsigma2=0.
                    else:
                        dsigma_grad_term = 0.0

                    # Combine all terms to get the complete gradient contribution
                    grad += dy_grad_term / sigma**2 + dsigma_grad_term

                snllh[par_opt_idx] = grad

        return snllh

    @staticmethod
    def get_default_options() -> dict:
        """Return default options for solving the inner problem."""
        options = {
            MIN_DIFF_FACTOR: 0.0,
            REGULARIZE_SPLINE: False,
            REGULARIZATION_FACTOR: 0.0,
        }
        return options

    def _optimize_spline(
        self,
        inner_parameters: list[SplineInnerParameter],
        group_dict: dict,
    ):
        """Run optimization for the inner problem.

        Parameters
        ----------
        inner_parameters:
            The spline inner parameters.
        group_dict:
            The group dictionary.
        """
        (
            distance_between_bases,
            spline_bases,
            intervals_per_sim,
        ) = self._rescale_spline_bases(
            sim_all=group_dict[CURRENT_SIMULATION],
            N=group_dict[N_SPLINE_PARS],
            K=group_dict[NUM_DATAPOINTS],
        )

        min_diff = self._get_minimal_difference(
            measurement_range=group_dict[MAX_DATAPOINT]
            - group_dict[MIN_DATAPOINT],
            N=group_dict[N_SPLINE_PARS],
            min_diff_factor=self.options[MIN_DIFF_FACTOR],
        )

        inner_options = self._get_inner_optimization_options(
            inner_parameters=inner_parameters,
            N=group_dict[N_SPLINE_PARS],
            min_meas=group_dict[MIN_DATAPOINT],
            max_meas=group_dict[MAX_DATAPOINT],
            min_diff=min_diff,
        )

        # Wrap the analytical optimization of sigma and
        # the regularization into the objective function
        def objective_function_wrapper(x):
            return _calculate_nllh_for_group(
                s=x,
                sim_all=group_dict[CURRENT_SIMULATION],
                measurements=group_dict[DATAPOINTS],
                N=group_dict[N_SPLINE_PARS],
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
                regularization_factor=self.options[REGULARIZATION_FACTOR],
                regularize_spline=self.options[REGULARIZE_SPLINE],
                group_dict=group_dict,
            )

        # Wrap the analytical optimization of sigma and
        # the regularization into the gradient function
        def inner_gradient_wrapper(x):
            return _calculate_nllh_gradient_for_group(
                s=x,
                sim_all=group_dict[CURRENT_SIMULATION],
                measurements=group_dict[DATAPOINTS],
                N=group_dict[N_SPLINE_PARS],
                delta_c=distance_between_bases,
                c=spline_bases,
                n=intervals_per_sim,
                regularization_factor=self.options[REGULARIZATION_FACTOR],
                regularize_spline=self.options[REGULARIZE_SPLINE],
                group_dict=group_dict,
            )

        results = minimize(
            objective_function_wrapper,
            jac=inner_gradient_wrapper,
            **inner_options,
        )

        return results

    @staticmethod
    def _rescale_spline_bases(sim_all: np.ndarray, N: int, K: int):
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
                        "Interval for a simulation has been set to a larger "
                        "value than the number of spline parameters.",
                        stacklevel=2,
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
        inner_parameters: list[SplineInnerParameter],
        N: int,
        min_meas: float,
        max_meas: float,
        min_diff: float,
    ) -> dict:
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
            "options": {"disp": None},
            "bounds": Bounds(lb=constraint_min_diff),
        }

        return inner_options


def _calculate_nllh_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
    regularization_factor: float,
    regularize_spline: bool,
    group_dict: dict,
) -> float:
    """Calculate the negative log-likelihood for the group.

    Combines the sum of squared residuals, the noise parameter,
    and the regularization term to the negative log-likelihood.

    Parameters
    ----------
    s:
        Reformulated inner spline parameters.
    sim_all:
        Simulations for the group.
    measurements:
        Measurements for the group.
    N:
        Number of spline bases.
    delta_c:
        Distance between two spline bases.
    c:
        Spline bases.
    n:
        Indices of the spline bases.
    regularization_factor:
        Regularization factor.
    regularize_spline:
        Whether to regularize the spline.
    group_dict:
        Dictionary containing the group information.

    Returns
    -------
    Negative log-likelihood.
    """
    # Calculate residuals
    residuals_squared = _calculate_residuals_for_group(
        s=s,
        sim_all=sim_all,
        measurements=measurements,
        N=N,
        delta_c=delta_c,
        c=c,
        n=n,
    )
    K = len(sim_all)

    # Calculate sigma
    if group_dict[OPTIMIZE_NOISE]:
        sigma = _calculate_sigma_for_group(
            residuals_squared=residuals_squared,
            n_datapoints=N,
        )
        group_dict[INNER_NOISE_PARS] = sigma
    else:
        sigma = group_dict[INNER_NOISE_PARS]

    # Calculate regularization term
    if regularize_spline:
        regularization_term = _calculate_regularization_for_group(
            s=s,
            N=N,
            c=c,
            regularization_factor=regularization_factor,
        )
    else:
        regularization_term = 0.0

    # Combine all terms into the negative log-likelihood
    nllh = (
        0.5 * np.log(2 * np.pi * sigma**2) * K
        + residuals_squared / (sigma**2)
        + regularization_term
    )
    return nllh


def _calculate_nllh_gradient_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
    regularization_factor: float,
    regularize_spline: bool,
    group_dict: dict,
) -> np.ndarray:
    """Calculate the gradient of the nllh wrt. spline differences s for the group.

    Combines the gradient of the sum of squared residuals and the gradient of the
    regularization term to the gradient of the negative log-likelihood.

    Parameters
    ----------
    s:
        Reformulated inner spline parameters.
    sim_all:
        Simulations for the group.
    measurements:
        Measurements for the group.
    N:
        Number of spline bases.
    delta_c:
        Distance between two spline bases.
    c:
        Spline bases.
    n:
        Indices of the spline bases.
    regularization_factor:
        Regularization factor.
    regularize_spline:
        Whether to regularize the spline.
    group_dict:
        Dictionary containing the group information.

    Returns
    -------
    Gradient of the negative log-likelihood wrt. spline differences s.
    """
    # Calculate gradient of residuals
    residuals_squared_gradient = _calculate_residuals_gradient_for_group(
        s=s,
        sim_all=sim_all,
        measurements=measurements,
        N=N,
        delta_c=delta_c,
        c=c,
        n=n,
    )

    # Calculate sigma
    if group_dict[OPTIMIZE_NOISE]:
        residuals_squared = _calculate_residuals_for_group(
            s=s,
            sim_all=sim_all,
            measurements=measurements,
            N=N,
            delta_c=delta_c,
            c=c,
            n=n,
        )
        sigma = _calculate_sigma_for_group(
            residuals_squared=residuals_squared,
            n_datapoints=N,
        )
        group_dict[INNER_NOISE_PARS] = sigma
    else:
        sigma = group_dict[INNER_NOISE_PARS]

    # Calculate gradient of regularization term
    if regularize_spline:
        regularization_term_gradient = (
            _calculate_regularization_gradient_for_group(
                s=s,
                N=N,
                c=c,
                regularization_factor=regularization_factor,
            )
        )
    else:
        regularization_term_gradient = np.zeros_like(s)

    # Combine all terms into the gradient of the negative log-likelihood
    nllh_gradient = (
        residuals_squared_gradient / (sigma**2) + regularization_term_gradient
    )
    return nllh_gradient


def _calculate_sigma_for_group(
    residuals_squared: float,
    n_datapoints: int,
):
    """Calculate the noise parameter sigma.

    Parameters
    ----------
    residuals_squared:
        The sum of squared residuals divided by 2.
    n_datapoints:
        The number of datapoints.
    """
    sigma = np.sqrt(2 * residuals_squared / n_datapoints)

    return sigma


def _calculate_residuals_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Residuals squared for reformulated inner spline problem.

    Equal to 1/2 * sum_k (tilde{z}_k - z_k)^2
    """
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


def _calculate_residuals_gradient_for_group(
    s: np.ndarray,
    sim_all: np.ndarray,
    measurements: np.ndarray,
    N: int,
    delta_c: float,
    c: np.ndarray,
    n: np.ndarray,
):
    """Gradient of the residuals with respect to the spline differences s_i for a group."""

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


def _calculate_regularization_for_group(
    s: np.ndarray,
    N: int,
    c: np.ndarray,
    regularization_factor: float,
):
    """Calculate regularization term the given spline.

    We regularize the spline to be linear. To do this, we calculate the optimal
    linear function that minimizes the sum of squared residuals with respect to
    the spline knots. Then we calculate the sum of squared residuals for this
    linear function. If the calculated offset is smaller than 0, we set it to 0.
    This is because the spline is not allowed to be negative.
    """
    # Calculate the spline knots xi_i from spline differences s_i
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

    # Calculate the sum of squared residuals for the optimal linear function
    regularization_term = np.sum((xi - alpha_opt * c - beta_opt) ** 2) / (
        2 * N
    )

    return regularization_term * regularization_factor


def _calculate_regularization_gradient_for_group(
    s: np.ndarray,
    N: int,
    c: np.ndarray,
    regularization_factor: float,
):
    """Calculate regularization term gradient for the given spline."""
    # Calculate the spline knots xi_i from spline differences s_i

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

    # If the offset is smaller than 0, we set it to 0.
    # Otherwise, we calculate the gradient of the offset.
    if beta_opt < 0:
        beta_opt = 0

    # Calculate the slope of the optimal linear function
    alpha_opt = (c_dot_xi - beta_opt * c_sum) / c_squares_sum

    # Calculate some more auxiliary values
    residuals = xi - alpha_opt * c - beta_opt

    # Can remove terms from this aux_matrix due to optimality
    # of the linear function (alpha & beta)
    aux_matrix = lower_trian

    # Calculate the gradient of the sum of squared residuals
    regularization_gradient = residuals @ aux_matrix / N

    return regularization_gradient * regularization_factor


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
    lower_trian = np.tril(np.ones((N, N)))
    xi = np.dot(lower_trian, s)

    for y_k, n_k, index in zip(sim_all, n, range(len(sim_all))):
        interval_index = n_k - 1
        if interval_index == 0 or interval_index == N:
            mapped_simulations[index] = xi[interval_index]
        else:
            mapped_simulations[index] = (y_k - c[interval_index - 1]) * (
                xi[interval_index] - xi[interval_index - 1]
            ) / delta_c + xi[interval_index - 1]

    return mapped_simulations


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
    expdata: list[np.ndarray], mask: list[np.ndarray]
):
    """Extract data from expdata list of arrays for the given mask."""
    return np.concatenate(
        [
            expdata[condition_index][mask[condition_index]]
            for condition_index in range(len(mask))
        ]
    )


def save_inner_parameters_to_inner_problem(
    inner_problem: SemiquantProblem,
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

    for idx in range(len(inner_spline_parameters)):
        inner_spline_parameters[idx].value = s[idx]

    sigma = group_dict[INNER_NOISE_PARS]

    if group_dict[OPTIMIZE_NOISE]:
        inner_noise_parameters[0].value = sigma


def get_monotonicity_measure(measurement, simulation):
    """Get monotonicity measure by calculating inversions.

    Calculates the number of inversions in the simulation data
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
