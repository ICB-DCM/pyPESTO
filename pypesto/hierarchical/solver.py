import numpy as np
import copy
from typing import Dict, List

from ..objective import Objective
from ..problem import Problem
from ..optimize import minimize, Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem, scale_value_dict


class InnerSolver:

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> Dict[str, float]:
        """Solve the subproblem"""


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


def compute_optimal_scaling(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        sigma: List[np.ndarray],
        mask: List[np.ndarray]) -> float:
    """Compute optimal scaling."""
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, sigma_i, mask_i in \
            zip(sim, data, sigma, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        sigma_x = sigma_i[mask_i]
        # update statistics
        num += np.nansum(sim_x * data_x / sigma_x ** 2)
        den += np.nansum(sim_x ** 2 / sigma_x ** 2)

    # compute optimal value
    x_opt = 1.0  # value doesn't matter
    if not np.isclose(den, 0.0):
        x_opt = num / den

    return float(x_opt)


def apply_scaling(
        scaling_value: float,
        sim: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sim)):
        sim[i][mask[i]] = scaling_value * sim[i][mask[i]]


def compute_optimal_sigma(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        mask: List[np.ndarray]) -> float:
    # numerator, denominator
    num, den = 0.0, 0.0

    # iterate over conditions
    for sim_i, data_i, mask_i in zip(sim, data, mask):
        # extract relevant values
        sim_x = sim_i[mask_i]
        data_x = data_i[mask_i]
        # update statistics
        num += np.nansum((data_x - sim_x) ** 2)
        den += sim_x.size

    # compute optimal value
    x_opt = 1.0  # value doesn't matter
    if not np.isclose(x_opt, 0.0):
        # we report the standard deviation, not the variance
        x_opt = np.sqrt(num / den)

    return float(x_opt)


def apply_sigma(
        sigma_value: float,
        sigma: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply scaling to simulations (in-place)."""
    for i in range(len(sigma)):
        sigma[i][mask[i]] = sigma_value * sigma[i][mask[i]]


def compute_nllh(
        data: List[np.ndarray],
        sim: List[np.ndarray],
        sigma: List[np.ndarray]) -> float:
    nllh = 0.0
    for data_i, sim_i, sigma_i in zip(data, sim, sigma):
        nllh += 0.5 * np.nansum(np.log(2*np.pi*sigma_i**2))
        nllh += 0.5 * np.nansum((data_i-sim_i)**2 / sigma_i**2)
    return nllh


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
        result = minimize(pypesto_problem, n_starts=1)

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
