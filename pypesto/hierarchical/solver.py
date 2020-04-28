import numpy as np
import copy
from typing import Dict, List

from ..objective import Objective
from ..problem import Problem
from ..optimize import minimize, Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem, scale_value_dict


REDUCED = 'reduced'
STANDARD = 'standard'


class InnerSolver:

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

        #compute optimal offsets
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


def compute_optimal_offset(
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
        num += np.nansum((data_x - sim_x) / sigma_x ** 2)
        den += np.nansum(1 / sigma_x ** 2)

    # compute optimal value
    x_opt = 0.0  # value doesn't matter
    if not np.isclose(den, 0.0):
        x_opt = num / den

    return float(x_opt)


def apply_offset(
        offset_value: float,
        sim: List[np.ndarray],
        mask: List[np.ndarray]):
    """Apply offset to simulations (in-place)."""
    for i in range(len(sim)):
        sim[i][mask[i]] = sim[i][mask[i]] + offset_value


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


class OptimalScalingInnerSolver(InnerSolver):
    """
    Solve the inner subproblem of the optimal scaling approach for ordinal data.
    """

    def __init__(self, n_starts: int = 1, n_records: int = 1,
                 optimizer: Optimizer = None, options: Dict = None):
        self.n_starts = n_starts
        self.n_records = n_records
        self.optimizer = optimizer
        self.options = options
        if self.options is None:
            self.options = OptimalScalingInnerSolver.get_default_options()

        self.x_guesses = None

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> list:
        pars = problem.xs.values()
        x_names = [x.id for x in pars]
        data = problem.data

        optimal_surrogate = compute_optimal_surrogate_data(problem, sim, self.options)

        return optimal_surrogate

    @staticmethod
    def calculate_obj_functon(x_inner_opt):
        obj = np.sum([x_inner_opt[idx]['fun'] for idx in range(len(x_inner_opt))])
        return obj

    @staticmethod
    def get_default_options():
        options = {'method': 'reduced', 'reparameterized': True, 'intervalConstraints': 'max'}
        return options


def compute_optimal_surrogate_data(problem, sim, options):
    # compute optimal surrogate data and return as list of edatas
    optimal_surrogates = []
    for gr in problem.get_groups_for_xs(InnerParameter.OPTIMALSCALING):
        xs = problem.get_xs_for_group(gr)
        if options['method'] == REDUCED:
            surrogate_opt_results = optimize_surrogate_data_reduced(xs, sim, options)
        elif options['method'] == STANDARD:
            raise NotImplementedError('Standard optimization not implemented yet')
            # surrogate_opt_results = optimize_surrogate_data_standard(xs, sim, problem, options)
        else:
            raise ValueError(f"Unknown method {options['method']} for surrogate data calculation.")
        # write_surrogate_to_edatas_reduced(surrogate_opt_results, xs, edatas, sim, problem)
        optimal_surrogates.append(surrogate_opt_results)
    return optimal_surrogates


def optimize_surrogate_data_reduced(xs, sim, options):
    from scipy.optimize import minimize

    interval_range, interval_gap = compute_interval_constraints(xs, sim, options)
    w = get_weight_for_surrogate(xs, sim)

    obj_surr = lambda x: obj_surrogate_data_reduced(xs, x, sim, interval_gap,
                                                    interval_range, w, options)

    inner_options = get_default_inner_options(options, xs, sim, interval_range, interval_gap)

    results = minimize(obj_surr, **inner_options)
    return results


def get_default_inner_options(options, xs, sim, interval_range, interval_gap):
    from scipy.optimize import Bounds

    parameter_length = len(xs)

    min_all, max_all = get_min_max(xs, sim)
    x0 = np.linspace(np.max([min_all, interval_range]), max_all + interval_range, parameter_length)

    if options['reparameterized']:
        x0 = y2xi(x0, xs, interval_gap, interval_range)
        bounds = Bounds([0] * parameter_length, [max_all] * parameter_length)

        inner_options = {'x0': x0, 'method': 'L-BFGS-B',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'bounds': bounds}
    else:
        constraints = get_constraints_for_optimization_reduced(xs, sim, options)

        inner_options = {'x0': x0, 'method': 'SLSQP',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'constraints': constraints}
    return inner_options


def get_min_max(xs, sim):
    min_all = np.inf
    max_all = -np.inf
    for x in xs:
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            sim_x = sim_i[mask_i]
            if mask_i.any():
                min_all = np.min([min_all, sim_x])
                max_all = np.max([max_all, sim_x])
    return min_all, max_all


def get_sim_all(xs, sim):
    return


def get_weight_for_surrogate(xs, sim):
    sim_x_all = []
    for x in xs:
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            if mask_i.any():
                sim_x = sim_i[mask_i]
                sim_x_all.append(sim_x)
    eps = 1e-10
    v_net = 0
    for idx in range(len(sim_x_all) - 1):
        v_net += np.abs(sim_x_all[idx + 1] - sim_x_all[idx])
    w = 0.5 * np.sum(np.abs(sim_x_all)) + v_net + eps
    return w ** 2


def compute_interval_constraints(xs, sim, options):
    # compute constraints on interval size and interval gap size similar to Pargett et al. (2014)
    eps = 1e-16
    min_simulation, max_simulation = get_min_max(xs, sim)

    if options['intervalConstraints'] == 'max-min':
        interval_range = (max_simulation - min_simulation) / (2 * len(xs) + 1)
        interval_gap = (max_simulation - min_simulation) / (4 * (len(xs) - 1) + 1)
    elif options['intervalConstraints'] == 'max':
        interval_range = max_simulation / (2 * len(xs) + 1)
        interval_gap = max_simulation / (4 * (len(xs) - 1) + 1)
    else:
        raise ValueError(f"intervalConstraints = {options['intervalConstraints']} not implemented.")
    if interval_gap < eps:
        interval_gap = eps
    return interval_range, interval_gap


def y2xi(optimal_scaling_bounds, xs, interval_gap, interval_range):
    optimal_scaling_bounds_reparameterized = \
        np.full(shape=(np.shape(optimal_scaling_bounds)), fill_value=np.nan)

    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            optimal_scaling_bounds_reparameterized[x_category - 1] = \
                optimal_scaling_bounds[x_category - 1]\
                - interval_range
        else:
            optimal_scaling_bounds_reparameterized[x_category - 1] = \
                optimal_scaling_bounds[x_category - 1]\
                - optimal_scaling_bounds[x_category - 2]\
                - interval_gap - interval_range

    return optimal_scaling_bounds_reparameterized


def xi2y(optimal_scaling_bounds_reparameterized, xs, interval_gap, interval_range):
    # TODO: optimal scaling parameters in parameter sheet have to be ordered at the moment
    optimal_scaling_bounds = np.full(shape=(np.shape(optimal_scaling_bounds_reparameterized)),
                                     fill_value=np.nan)
    for x in xs:
        x_category = int(x.category)
        if x_category == 1:
            optimal_scaling_bounds[x_category - 1] = \
                interval_range + optimal_scaling_bounds_reparameterized[
                x_category - 1]
        else:
            optimal_scaling_bounds[x_category - 1] = \
                optimal_scaling_bounds_reparameterized[x_category - 1] +\
                interval_gap + interval_range + optimal_scaling_bounds[
                    x_category - 2]
    return optimal_scaling_bounds


def obj_surrogate_data_reduced(xs, optimal_scaling_bounds, sim, interval_gap,
                               interval_range, w, options):
    # compute optimal scaling objective function
    obj = 0.0
    y_sim_all = []
    y_surrogate_all = []
    if options['reparameterized']:
        optimal_scaling_bounds = xi2y(optimal_scaling_bounds, xs, interval_gap, interval_range)

    for x in xs:
        x_category = int(x.category)
        x_upper = optimal_scaling_bounds[x_category - 1]
        if x_category == 1:
            x_lower = 0
        elif x_category > 1:
            x_lower = optimal_scaling_bounds[x_category - 2] + interval_gap
        else:
            raise ValueError('Category value needs to be larger than 0.')
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            if mask_i.any():
                y_sim = sim_i[mask_i]
                if x_lower > y_sim:
                    y_surrogate = x_lower
                elif y_sim > x_upper:
                    y_surrogate = x_upper
                else:
                    y_surrogate = y_sim
                obj += (y_surrogate - y_sim) ** 2
                y_sim_all.append(y_sim)
                y_surrogate_all.append(y_surrogate)
    obj = np.divide(obj, w)
    return obj


def get_constraints_for_optimization_reduced(xs, sim, options):
    # TODO
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(xs, sim, options)
    # A = np.diag(-np.ones(num_categories)) + np.diag(np.ones(num_categories-1),1)
    # A = A[:-1, :]

    A = np.diag(-np.ones(num_categories), -1) + np.diag(np.ones(num_categories + 1))
    A = A[:-1, :-1]
    b = np.empty((num_categories,))
    b[0] = interval_range
    b[1:] = interval_range + interval_gap
    ineq_cons = {'type': 'ineq', 'fun': lambda x: A.dot(x) - b}

    # from scipy.optimize import LinearConstraint
    # linear_constraint = LinearConstraint(A, b, [np.inf]*len(b))
    return ineq_cons
    # return linear_constraint

# def optimize_surrogate_data_standard(xs, sim, problem, options):
