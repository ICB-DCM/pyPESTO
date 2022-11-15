import copy
from typing import Any, Dict, List

import numpy as np

from ..C import InnerParameterType
from ..objective import Objective
from ..optimize import minimize
from ..problem import Problem
from .problem import InnerProblem, scale_value_dict
from .util import (
    apply_offset,
    apply_scaling,
    apply_sigma,
    compute_nllh,
    compute_optimal_offset,
    compute_optimal_offset_coupled,
    compute_optimal_scaling,
    compute_optimal_sigma,
)


class InnerSolver:
    """Solver for an inner optimization problem."""

    def initialize(self):
        """
        (Re-)initialize the solver.

        Default: Do nothing.
        """

    def solve(
        self,
        problem: InnerProblem,
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        scaled: bool,
    ) -> Dict[str, float]:
        """Solve the subproblem.

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


class AnalyticalInnerSolver(InnerSolver):
    """Solve the inner subproblem analytically.

    Currently, supports offset and scaling parameters (coupled or not), and
    sigmas for additive Gaussian noise.
    """

    def solve(
        self,
        problem: InnerProblem,
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        scaled: bool,
    ) -> Dict[str, float]:
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

        # compute optimal offsets
        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            if x.coupled:
                x_opt[x.inner_parameter_id] = compute_optimal_offset_coupled(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )
            else:
                x_opt[x.inner_parameter_id] = compute_optimal_offset(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )
        # apply offsets
        for x in problem.get_xs_for_type(InnerParameterType.OFFSET):
            apply_offset(
                offset_value=x_opt[x.inner_parameter_id], data=data, mask=x.ixs
            )

        # compute optimal scalings
        for x in problem.get_xs_for_type(InnerParameterType.SCALING):
            x_opt[x.inner_parameter_id] = compute_optimal_scaling(
                data=data, sim=sim, sigma=sigma, mask=x.ixs
            )
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


class NumericalInnerSolver(InnerSolver):
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
        minimize_kwargs: Dict[str, Any] = None,
        n_cached: int = 1,
        problem_kwargs: Dict[str, Any] = None,
    ):
        self.minimize_kwargs = minimize_kwargs
        if self.minimize_kwargs is None:
            self.minimize_kwargs = {}
        self.n_cached = n_cached
        self.problem_kwargs = problem_kwargs
        if self.problem_kwargs is None:
            self.problem_kwargs = {}

        self.minimize_kwargs['n_starts'] = self.minimize_kwargs.get(
            'n_starts', 1
        )
        self.minimize_kwargs['progress_bar'] = self.minimize_kwargs.get(
            'progress_bar', False
        )

        self.x_guesses = None
        self.dummy_lb = -1e20
        self.dummy_ub = +1e20

    def initialize(self):
        """(Re-)initialize the solver."""
        self.x_guesses = None

    def solve(
        self,
        problem: InnerProblem,
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        scaled: bool,
    ) -> Dict[str, float]:
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
        pars = problem.xs.values()
        # We currently cannot handle constraints on inner parameters correctly,
        # and would have to assume [-inf, inf]. However, this may not be
        # supported by all inner optimizers, so we go for some (arbitrary)
        # large value.
        lb = np.array(
            [
                0
                if x.inner_parameter_type == InnerParameterType.SIGMA
                else self.dummy_lb
                for x in pars
            ]
        )

        ub = np.full(shape=len(pars), fill_value=self.dummy_ub)

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

            return compute_nllh(_data, _sim, _sigma)

        # TODO gradient
        objective = Objective(fun)

        # optimization problem
        pypesto_problem = Problem(
            objective, lb=lb, ub=ub, x_names=x_names, **self.problem_kwargs
        )

        if self.x_guesses is not None:
            pypesto_problem.set_x_guesses(
                self.x_guesses[:, pypesto_problem.x_free_indices]
            )
        else:
            pypesto_problem.set_x_guesses(
                [list(problem.get_dummy_values(scaled=False).values())]
            )

        # perform the actual optimization
        result = minimize(pypesto_problem, **self.minimize_kwargs)

        best_par = result.optimize_result.list[0]['x']

        if (np.isclose(best_par, lb) | np.isclose(best_par, ub)).any():
            raise RuntimeError(
                "Active bounds in inner problem optimization. This can result "
                "in incorrect gradient computation for the outer parameters."
            )

        x_opt = dict(zip(pypesto_problem.x_names, best_par))

        # cache
        self.x_guesses = np.array(
            [
                entry['x']
                for entry in result.optimize_result.list[: self.n_cached]
            ]
        )

        # scale
        if scaled:
            x_opt = scale_value_dict(x_opt, problem)

        return x_opt
