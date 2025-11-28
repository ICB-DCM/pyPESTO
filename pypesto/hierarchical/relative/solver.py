from __future__ import annotations

import copy
from typing import Any

import numpy as np

from ...C import InnerParameterType
from ...objective import Objective
from ...objective.amici.amici_util import add_sim_grad_to_opt_grad
from ...optimize import minimize
from ...problem import Problem
from ..base_parameter import InnerParameter
from ..base_problem import (
    InnerProblem,
    scale_back_value_dict,
    scale_value_dict,
)
from ..base_solver import InnerSolver
from .util import (
    apply_offset,
    apply_scaling,
    apply_scaling_to_sensitivities,
    apply_sigma,
    compute_bounded_optimal_scaling_offset_coupled,
    compute_nllh,
    compute_nllh_gradient_for_condition,
    compute_optimal_offset,
    compute_optimal_offset_coupled,
    compute_optimal_scaling,
    compute_optimal_sigma,
)

try:
    import amici
    from amici.petab.parameter_mapping import ParameterMapping
except ImportError:
    pass


class RelativeInnerSolver(InnerSolver):
    """Base class for the relative inner solver."""

    def calculate_obj_function(
        self,
        problem: InnerProblem,
        sim: list[np.ndarray],
        sigma: list[np.ndarray],
        inner_parameters: dict[str, float],
    ) -> float:
        """Calculate the objective function value.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        sim:
            List of model output matrices, as provided in AMICI's
            ``ReturnData.y``. Same order as simulations in the
            PEtab problem.
        sigma:
            List of sigma matrices from the model, as provided in AMICI's
            ``ReturnData.sigmay``. Same order as simulations in the
            PEtab problem.
        """
        relevant_data = copy.deepcopy(problem.data)
        sim = copy.deepcopy(sim)
        sigma = copy.deepcopy(sigma)
        inner_parameters = copy.deepcopy(inner_parameters)
        inner_parameters = scale_back_value_dict(inner_parameters, problem)

        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            apply_offset(
                offset_value=inner_parameters[x.inner_parameter_id],
                data=relevant_data,
                mask=x.ixs,
            )

        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            apply_scaling(
                scaling_value=inner_parameters[x.inner_parameter_id],
                sim=sim,
                mask=x.ixs,
            )

        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            apply_sigma(
                sigma_value=inner_parameters[x.inner_parameter_id],
                sigma=sigma,
                mask=x.ixs,
            )

        return compute_nllh(relevant_data, sim, sigma, problem.data_mask)

    def calculate_gradients(
        self,
        problem: InnerProblem,
        sim: list[np.ndarray],
        ssim: list[np.ndarray],
        sigma: list[np.ndarray],
        ssigma: list[np.ndarray],
        inner_parameters: dict[str, float],
        parameter_mapping: ParameterMapping,
        par_opt_ids: list[str],
        par_sim_ids: list[str],
        snllh: np.ndarray,
    ) -> np.ndarray:
        """Calculate the gradients with respect to the outer parameters.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        sim:
            List of model output matrices, as provided in AMICI's
            ``ReturnData.y``. Same order as simulations in the
            PEtab problem.
        ssim:
            List of sensitivity matrices from the model, as provided in AMICI's
            ``ReturnData.sy``. Same order as simulations in the
            PEtab problem.
        sigma:
            List of sigma matrices from the model, as provided in AMICI's
            ``ReturnData.sigmay``. Same order as simulations in the
            PEtab problem.
        ssigma:
            List of sigma sensitivity matrices from the model, as provided
            in AMICI's ``ReturnData.ssigmay``. Same order as simulations in the
            PEtab problem.
        inner_parameters:
            The computed inner parameters.
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
        relevant_data = copy.deepcopy(problem.data)
        sim = copy.deepcopy(sim)
        sigma = copy.deepcopy(sigma)
        inner_parameters = copy.deepcopy(inner_parameters)
        inner_parameters = scale_back_value_dict(inner_parameters, problem)

        # restructure sensitivities to have parameter index as second index
        ssim = [
            np.asarray(
                [
                    ssim[cond_idx][:, par_idx, :]
                    for par_idx in range(ssim[cond_idx].shape[1])
                ]
            )
            for cond_idx in range(len(sim))
        ]
        ssigma = [
            np.asarray(
                [
                    ssigma[cond_idx][:, par_idx, :]
                    for par_idx in range(ssigma[cond_idx].shape[1])
                ]
            )
            for cond_idx in range(len(sigma))
        ]

        # apply offsets, scalings and sigmas
        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            apply_offset(
                offset_value=inner_parameters[x.inner_parameter_id],
                data=relevant_data,
                mask=x.ixs,
            )

        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            apply_scaling(
                scaling_value=inner_parameters[x.inner_parameter_id],
                sim=sim,
                mask=x.ixs,
            )
            apply_scaling_to_sensitivities(
                scaling_value=inner_parameters[x.inner_parameter_id],
                ssim=ssim,
                mask=x.ixs,
            )

        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            apply_sigma(
                sigma_value=inner_parameters[x.inner_parameter_id],
                sigma=sigma,
                mask=x.ixs,
            )

        # compute gradients
        for cond_idx, cond_par_map in enumerate(parameter_mapping):
            gradient_for_cond = compute_nllh_gradient_for_condition(
                data=relevant_data[cond_idx],
                sim=sim[cond_idx],
                ssim=ssim[cond_idx],
                sigma=sigma[cond_idx],
                ssigma=ssigma[cond_idx],
            )
            add_sim_grad_to_opt_grad(
                par_opt_ids=par_opt_ids,
                par_sim_ids=par_sim_ids,
                condition_map_sim_var=cond_par_map.map_sim_var,
                sim_grad=gradient_for_cond,
                opt_grad=snllh,
            )

        return snllh

    def apply_inner_parameters_to_rdatas(
        self,
        problem: InnerProblem,
        rdatas: list[amici.ReturnData],
        inner_parameters: dict[str, float],
    ):
        """Apply the inner parameters to the rdatas.

        If we have simulated the model with dummy inner parameters only,
        we need to apply the inner parameters to the rdatas to get the
        correct model output.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        rdatas:
            The rdatas to apply the inner parameters to.
        inner_parameters:
            The inner parameters to apply to the rdatas.
        """
        sim = [rdata["y"] for rdata in rdatas]
        sigma = [rdata["sigmay"] for rdata in rdatas]
        inner_parameters = copy.deepcopy(inner_parameters)
        inner_parameters = scale_back_value_dict(inner_parameters, problem)

        # apply offsets, scalings and sigmas
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            apply_scaling(
                scaling_value=inner_parameters[x.inner_parameter_id],
                sim=sim,
                mask=x.ixs,
            )

        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            apply_offset(
                offset_value=inner_parameters[x.inner_parameter_id],
                data=sim,
                mask=x.ixs,
                is_data=False,
            )

        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            apply_sigma(
                sigma_value=inner_parameters[x.inner_parameter_id],
                sigma=sigma,
                mask=x.ixs,
            )

        return rdatas


class AnalyticalInnerSolver(RelativeInnerSolver):
    """Solve the inner subproblem analytically.

    Currently, supports sigmas for additive Gaussian noise.
    """

    def solve(
        self,
        problem: InnerProblem,
        sim: list[np.ndarray],
        sigma: list[np.ndarray],
        scaled: bool,
    ) -> dict[str, float]:
        """Solve the subproblem analytically.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        sim:
            List of model output matrices, as provided in AMICI's
            ``ReturnData.y``. Same order as simulations in the
            PEtab problem.
        sigma:
            List of sigma matrices from the model, as provided in AMICI's
            ``ReturnData.sigmay``. Same order as simulations in the
            PEtab problem.
        scaled:
            Whether to scale the results to the parameter scale specified in
            ``problem``.
        """
        x_opt = {}
        data = copy.deepcopy(problem.data)
        sim = copy.deepcopy(sim)
        sigma = copy.deepcopy(sigma)

        # compute optimal offsets
        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            if x.coupled is not None:
                x_opt[x.inner_parameter_id] = compute_optimal_offset_coupled(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )

                # calculate the optimal coupled scaling
                coupled_scaling = x.coupled
                x_opt[coupled_scaling.inner_parameter_id] = (
                    compute_optimal_scaling(
                        data=data,
                        sim=sim,
                        sigma=sigma,
                        mask=coupled_scaling.ixs,
                        optimal_offset=x_opt[x.inner_parameter_id],
                    )
                )

                # check whether they both satisfy their bounds
                if x.is_within_bounds(
                    x_opt[x.inner_parameter_id]
                ) and coupled_scaling.is_within_bounds(
                    x_opt[coupled_scaling.inner_parameter_id]
                ):
                    continue
                else:
                    # if not, we need to recompute them
                    (
                        x_opt[coupled_scaling.inner_parameter_id],
                        x_opt[x.inner_parameter_id],
                    ) = compute_bounded_optimal_scaling_offset_coupled(
                        data=data,
                        sim=sim,
                        sigma=sigma,
                        s=coupled_scaling,
                        b=x,
                        s_opt_value=x_opt[coupled_scaling.inner_parameter_id],
                        b_opt_value=x_opt[x.inner_parameter_id],
                    )
            # compute non-coupled optimal offset
            else:
                x_opt[x.inner_parameter_id] = compute_optimal_offset(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )
                # check if the solution is within bounds
                # if not, we set it to the unsatisfied bound
                if not x.is_within_bounds(x_opt[x.inner_parameter_id]):
                    x_opt[x.inner_parameter_id] = x.get_bounds()[
                        x.get_unsatisfied_bound(x_opt[x.inner_parameter_id])
                    ]

        # apply offsets
        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            apply_offset(
                offset_value=x_opt[x.inner_parameter_id], data=data, mask=x.ixs
            )

        # compute non-coupled optimal scalings
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            if x.coupled is None:
                x_opt[x.inner_parameter_id] = compute_optimal_scaling(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )
                # check if the solution is within bounds
                # if not, we set it to the unsatisfied bound
                if not x.is_within_bounds(x_opt[x.inner_parameter_id]):
                    x_opt[x.inner_parameter_id] = x.get_bounds()[
                        x.get_unsatisfied_bound(x_opt[x.inner_parameter_id])
                    ]
        # apply scalings
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            apply_scaling(
                scaling_value=x_opt[x.inner_parameter_id], sim=sim, mask=x.ixs
            )

        # compute optimal sigmas
        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            x_opt[x.inner_parameter_id] = compute_optimal_sigma(
                data=data, sim=sim, mask=x.ixs
            )
        # apply sigmas
        for x in problem.get_xs_for_type(InnerParameterType.SIGMA):
            apply_sigma(
                sigma_value=x_opt[x.inner_parameter_id],
                sigma=sigma,
                mask=x.ixs,
            )

        # scale
        if scaled:
            x_opt = scale_value_dict(x_opt, problem)
        return x_opt


class NumericalInnerSolver(RelativeInnerSolver):
    """Solve the inner subproblem numerically.

    Advantage: The structure of the subproblem does not matter like, at all.
    Disadvantage: Slower.

    Special features: We cache the best parameters, which substantially
    speeds things up.

    Attributes
    ----------
    minimize_kwargs:
        Passed to the `pypesto.optimize.minimize` call.
    n_cached:
        Number of optimized parameter vectors to save.
    problem_kwargs:
        Passed to the `pypesto.Problem` constructor.
    x_guesses:
        Cached optimized parameter vectors, supplied as guesses to the next
        `solve` call.
    """

    def __init__(
        self,
        minimize_kwargs: dict[str, Any] = None,
        n_cached: int = 1,
        problem_kwargs: dict[str, Any] = None,
    ):
        self.minimize_kwargs = minimize_kwargs
        if self.minimize_kwargs is None:
            self.minimize_kwargs = {}
        self.n_cached = n_cached
        self.problem_kwargs = problem_kwargs
        if self.problem_kwargs is None:
            self.problem_kwargs = {}

        self.minimize_kwargs["n_starts"] = self.minimize_kwargs.get(
            "n_starts", 1
        )
        self.minimize_kwargs["progress_bar"] = self.minimize_kwargs.get(
            "progress_bar", False
        )

        self.x_guesses = None
        self.dummy_lb = -1e20
        self.dummy_ub = +1e20
        self.user_specified_lb = None
        self.user_specified_ub = None

    def initialize(self):
        """(Re-)initialize the solver."""
        self.x_guesses = None

    def solve(
        self,
        problem: InnerProblem,
        sim: list[np.ndarray],
        sigma: list[np.ndarray],
        scaled: bool,
    ) -> dict[str, float]:
        """Solve the subproblem numerically.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        sim:
            List of model output matrices, as provided in AMICI's
            ``ReturnData.y``. Same order as simulations in the
            PEtab problem.
        sigma:
            List of sigma matrices from the model, as provided in AMICI's
            ``ReturnData.sigmay``. Same order as simulations in the
            PEtab problem.
        scale:
            Whether to scale the results to the parameter scale specified in
            ``problem``.
        """
        pars = list(problem.xs.values())

        # This has to be done only once
        if self.user_specified_lb is None or self.user_specified_ub is None:
            self.user_specified_lb = [
                i for i in range(len(pars)) if pars[i].lb != -np.inf
            ]
            self.user_specified_ub = [
                i for i in range(len(pars)) if pars[i].ub != np.inf
            ]

        lb = [x.lb for x in pars]
        ub = [x.ub for x in pars]

        x_guesses = self.sample_startpoints(problem, pars)

        x_names = [x.inner_parameter_id for x in pars]
        data = problem.data

        # objective function
        def fun(x):
            _sim = copy.deepcopy(sim)
            _sigma = copy.deepcopy(sigma)
            _data = copy.deepcopy(data)
            for x_val, par in zip(x, pars):
                mask = par.ixs
                if par.inner_parameter_type == InnerParameterType.OFFSET:
                    apply_offset(x_val, _data, mask)
                elif par.inner_parameter_type == InnerParameterType.SCALING:
                    apply_scaling(x_val, _sim, mask)
                elif par.inner_parameter_type == InnerParameterType.SIGMA:
                    apply_sigma(x_val, _sigma, mask)
                else:
                    raise ValueError(
                        "Can't handle parameter type "
                        f"`{par.inner_parameter_type}`."
                    )

            return compute_nllh(_data, _sim, _sigma, problem.data_mask)

        # TODO gradient
        objective = Objective(fun)

        # optimization problem
        pypesto_problem = Problem(
            objective, lb=lb, ub=ub, x_names=x_names, **self.problem_kwargs
        )
        pypesto_problem.set_x_guesses(
            x_guesses[:, pypesto_problem.x_free_indices]
        )

        # perform the actual optimization
        result = minimize(pypesto_problem, **self.minimize_kwargs)
        best_par = result.optimize_result.list[0]["x"]

        # Check if the index of an optimized parameter on the dummy bound
        # is not in the list of specified bounds. If so, raise an error.
        if any(
            (
                i not in self.user_specified_lb
                for i, x in enumerate(best_par)
                if x == self.dummy_lb
            )
        ) or any(
            (
                i not in self.user_specified_ub
                for i, x in enumerate(best_par)
                if x == self.dummy_ub
            )
        ):
            raise RuntimeError(
                f"An optimal inner parameter is on the default dummy bound of numerical optimization. "
                f"This means the optimal inner parameter is either extremely large (>={self.dummy_ub})"
                f"or extremely small (<={self.dummy_lb}). Consider changing the inner parameter bounds."
            )

        x_opt = dict(zip(pypesto_problem.x_names, best_par))

        # cache
        self.x_guesses = np.array(
            [
                entry["x"]
                for entry in result.optimize_result.list[: self.n_cached]
            ]
        )

        # scale
        if scaled:
            x_opt = scale_value_dict(x_opt, problem)

        return x_opt

    def sample_startpoints(
        self, problem: InnerProblem, pars: list[InnerParameter]
    ) -> np.ndarray:
        """Sample startpoints for the numerical optimization.

        Samples the startpoints for the numerical optimization from a
        log-uniform distribution using the symmetric logarithmic scale.

        Parameters
        ----------
        problem:
            The inner problem to solve.
        pars:
            The inner parameters to sample startpoints for.

        Returns
        -------
        The sampled startpoints appended to the cached startpoints.
        """
        if self.minimize_kwargs["n_starts"] == 1 and self.x_guesses is None:
            return np.array(
                [list(problem.get_dummy_values(scaled=False).values())]
            )
        elif self.x_guesses is not None:
            n_samples = self.minimize_kwargs["n_starts"] - len(self.x_guesses)
        else:
            n_samples = self.minimize_kwargs["n_starts"] - 1

        if n_samples <= 0:
            return self.x_guesses

        lb = np.nan_to_num([x.lb for x in pars], neginf=self.dummy_lb)
        ub = np.nan_to_num([x.ub for x in pars], posinf=self.dummy_ub)

        def symlog10(x):
            return np.sign(x) * np.log10(np.abs(x) + 1)

        def inverse_symlog10(x):
            return np.sign(x) * (np.power(10, np.abs(x)) - 1)

        # Sample startpoints from a log-uniform distribution
        startpoints = np.random.uniform(
            low=symlog10(lb),
            high=symlog10(ub),
            size=(n_samples, len(pars)),
        )
        startpoints = inverse_symlog10(startpoints)

        # Stack the sampled startpoints with the cached startpoints
        if self.x_guesses is not None:
            startpoints = np.vstack(
                (
                    self.x_guesses,
                    startpoints,
                )
            )
        else:
            startpoints = np.vstack(
                (
                    np.array(
                        [list(problem.get_dummy_values(scaled=False).values())]
                    ),
                    startpoints,
                )
            )
        return startpoints
