import numpy as np
import copy

from typing import Dict, List

from ..objective import Objective
from ..problem import Problem
from ..optimize import minimize, Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem, scale_value_dict
from .util import (
    compute_optimal_offset, apply_offset, compute_optimal_scaling,
    apply_scaling, compute_optimal_sigma, apply_sigma, compute_nllh)


class InnerSolver:

    def initialize(self):
        """Initialize the solver. Default: Do nothing."""

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> Dict[str, float]:
        """Solve the subproblem."""


class AnalyticalInnerSolver(InnerSolver):
    """Solve the inner subproblem analytically.

    Currently supports scalings and sigmas for additive Gaussian noise.
    """

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> Dict[str, float]:
        x_opt = {}

        # compute optimal offsets
        for x in problem.get_xs_for_type(InnerParameter.OFFSET):
            x_opt[x.id] = compute_optimal_offset(
                data=problem.data, sim=sim, sigma=sigma, mask=x.ixs)
        # apply offsets
        for x in problem.get_xs_for_type(InnerParameter.OFFSET):
            apply_offset(offset_value=x_opt[x.id], sim=sim, mask=x.ixs)

        # compute optimal scalings
        for x in problem.get_xs_for_type(InnerParameter.SCALING):
            x_opt[x.id] = compute_optimal_scaling(
                data=problem.data, sim=sim, sigma=sigma, mask=x.ixs)
        # apply scalings (TODO not always necessary)
        for x in problem.get_xs_for_type(InnerParameter.SCALING):
            apply_scaling(scaling_value=x_opt[x.id], sim=sim, mask=x.ixs)

        # compute optimal sigmas
        for x in problem.get_xs_for_type(InnerParameter.SIGMA):
            x_opt[x.id] = compute_optimal_sigma(
                data=problem.data, sim=sim, mask=x.ixs)
        # apply sigmas
        for x in problem.get_xs_for_type(InnerParameter.SIGMA):
            apply_sigma(sigma_value=x_opt[x.id], sigma=sigma, mask=x.ixs)

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
    """

    def __init__(self, n_starts: int = 1, n_records: int = 1,
                 optimizer: Optimizer = None):
        self.n_starts = n_starts
        self.n_records = n_records
        self.optimizer = optimizer

        self.x_guesses = None

    def initialize(self):
        self.x_guesses = None

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> Dict[str, float]:
        pars = problem.xs.values()
        lb = np.array([x.lb for x in pars])
        ub = np.array([x.ub for x in pars])
        x_names = [x.id for x in pars]
        data = problem.data

        # objective function
        def fun(x):
            _sim = copy.deepcopy(sim)
            _sigma = copy.deepcopy(sigma)
            for x_val, par in zip(x, pars):
                mask = par.ixs
                if par.type == InnerParameter.SCALING:
                    apply_scaling(x_val, _sim, mask)
                elif par.type == InnerParameter.OFFSET:
                    apply_offset(x_val, _sim, mask)
                elif par.type == InnerParameter.SIGMA:
                    apply_sigma(x_val, _sigma, mask)
                else:
                    raise ValueError("Can't handle parameter type.")
            nllh = compute_nllh(data, _sim, _sigma)
            return nllh
        # TODO gradient
        objective = Objective(fun)

        # optimization problem
        pypesto_problem = Problem(objective, lb=lb, ub=ub, x_names=x_names)

        if self.x_guesses is not None:
            pypesto_problem.x_guesses = self.x_guesses

        # perform the actual optimization
        result = minimize(pypesto_problem, n_starts=self.n_starts)
        #print(result.optimize_result.get_for_key('fval'))
        #print(result.optimize_result.get_for_key('id'))

        best_par = result.optimize_result.list[0]['x']
        x_opt = {x_name: val
                 for x_name, val in zip(pypesto_problem.x_names, best_par)}

        # cache
        self.x_guesses = np.array([
            entry['x']
            for entry in result.optimize_result.list[:self.n_records]])

        # scale
        if scaled:
            x_opt = scale_value_dict(x_opt, problem)

        return x_opt
