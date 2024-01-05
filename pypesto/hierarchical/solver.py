import copy
from typing import Any, Dict, List

import numpy as np

from ..C import InnerParameterType
from ..objective import Objective
from ..optimize import minimize
from ..problem import Problem
from .parameter import InnerParameter
from .problem import InnerProblem, scale_value_dict
from .util import (
    apply_offset,
    apply_scaling,
    apply_sigma,
    compute_bounded_optimal_scaling_offset_coupled,
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
            if x.coupled is not None:
                x_opt[x.inner_parameter_id] = compute_optimal_offset_coupled(
                    data=data, sim=sim, sigma=sigma, mask=x.ixs
                )

                # calculate the optimal coupled scaling
                coupled_scaling = x.coupled
                x_opt[
                    coupled_scaling.inner_parameter_id
                ] = compute_optimal_scaling(
                    data=data,
                    sim=sim,
                    sigma=sigma,
                    mask=coupled_scaling.ixs,
                    optimal_offset=x_opt[x.inner_parameter_id],
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
        self.user_specified_lb = None
        self.user_specified_ub = None

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

            return compute_nllh(_data, _sim, _sigma)

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
        best_par = result.optimize_result.list[0]['x']

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
                entry['x']
                for entry in result.optimize_result.list[: self.n_cached]
            ]
        )

        # scale
        if scaled:
            x_opt = scale_value_dict(x_opt, problem)

        return x_opt

    def sample_startpoints(
        self, problem: InnerProblem, pars: List[InnerParameter]
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
        if self.minimize_kwargs['n_starts'] == 1 and self.x_guesses is None:
            return np.array(
                [list(problem.get_dummy_values(scaled=False).values())]
            )
        elif self.x_guesses is not None:
            n_samples = self.minimize_kwargs['n_starts'] - len(self.x_guesses)
        else:
            n_samples = self.minimize_kwargs['n_starts'] - 1

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
