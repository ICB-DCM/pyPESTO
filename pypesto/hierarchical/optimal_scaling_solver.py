import numpy as np
from typing import Dict, List

from ..optimize import Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem
from .solver import InnerSolver

REDUCED = 'reduced'
STANDARD = 'standard'


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
        if self.options['method'] == STANDARD and self.options['reparameterized']:
            raise NotImplementedError('Combining standard approach with reparameterization not implemented.')
        self.x_guesses = None

    def solve(
            self,
            problem: InnerProblem,
            sim: List[np.ndarray],
            sigma: List[np.ndarray],
            scaled: bool,
    ) -> list:
        # pars = problem.xs.values()
        # x_names = [x.id for x in pars]
        # data = problem.data

        optimal_surrogate = compute_optimal_surrogate_data(problem, sim, self.options)

        return optimal_surrogate

    @staticmethod
    def calculate_obj_function(x_inner_opt):
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
        surrogate_opt_results = optimize_surrogate_data(xs, sim, options)
        # write_surrogate_to_edatas_reduced(surrogate_opt_results, xs, edatas, sim, problem)
        optimal_surrogates.append(surrogate_opt_results)
    return optimal_surrogates


def optimize_surrogate_data(xs, sim, options):
    from scipy.optimize import minimize

    interval_range, interval_gap = compute_interval_constraints(xs, sim, options)
    w = get_weight_for_surrogate(xs, sim)

    obj_surr = lambda x: obj_surrogate_data(xs, x, sim, interval_gap,
                                            interval_range, w, options)

    inner_options = get_inner_options(options, xs, sim, interval_range, interval_gap)

    results = minimize(obj_surr, **inner_options)
    return results


def get_inner_options(options, xs, sim, interval_range, interval_gap):
    from scipy.optimize import Bounds

    min_all, max_all = get_min_max(xs, sim)
    if options['method'] == REDUCED:
        parameter_length = len(xs)
        x0 = np.linspace(np.max([min_all, interval_range]), max_all + interval_range, parameter_length)
    elif options['method'] == STANDARD:
        parameter_length = 2 * len(xs)
        x0 = np.linspace(0, max_all + interval_range, parameter_length)
    else:
        raise NotImplementedError(f"Unkown optimal scaling method {options['method']}")

    if options['reparameterized']:
        x0 = y2xi(x0, xs, interval_gap, interval_range)
        bounds = Bounds([0] * parameter_length, [max_all] * parameter_length)

        inner_options = {'x0': x0, 'method': 'L-BFGS-B',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'bounds': bounds}
    else:
        constraints = get_constraints_for_optimization(xs, sim, options)

        inner_options = {'x0': x0, 'method': 'SLSQP',
                         'options': {'maxiter': 2000, 'ftol': 1e-10},
                         'constraints': constraints}
    return inner_options


def get_min_max(xs, sim):
    sim_all = get_sim_all(xs, sim)

    min_all = np.min(sim_all)
    max_all = np.max(sim_all)

    return min_all, max_all


def get_sim_all(xs, sim):
    sim_all = []
    for x in xs:
        for sim_i, mask_i in \
                zip(sim, x.ixs):
            sim_x = sim_i[mask_i]
            if mask_i.any():
                sim_all.append(sim_x[0])
    return sim_all


def get_weight_for_surrogate(xs, sim):
    sim_x_all = get_sim_all(xs, sim)
    eps = 1e-10
    v_net = 0
    for idx in range(len(sim_x_all) - 1):
        v_net += np.abs(sim_x_all[idx + 1] - sim_x_all[idx])
    w = 0.5 * np.sum(np.abs(sim_x_all)) + v_net + eps
    return w ** 2


def compute_interval_constraints(xs, sim, options):
    # compute constraints on interval size and interval gap size similar to Pargett et al. (2014)
    if 'minGap' not in options:
        eps = 1e-16
    else:
        eps = options['minGap']

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


def obj_surrogate_data(xs, optimal_scaling_bounds, sim, interval_gap,
                       interval_range, w, options):
    # compute optimal scaling objective function
    obj = 0.0
    if options['reparameterized']:
        optimal_scaling_bounds = xi2y(optimal_scaling_bounds, xs, interval_gap, interval_range)

    for x in xs:
        x_upper, x_lower = get_bounds_for_category(x, optimal_scaling_bounds, interval_gap, options)
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
    obj = np.divide(obj, w)
    return obj


def get_bounds_for_category(x, optimal_scaling_bounds, interval_gap, options):
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
        raise NotImplementedError(f"Unkown optimal scaling method {options['method']}")
    return x_upper, x_lower


def get_constraints_for_optimization(xs, sim, options):
    num_categories = len(xs)
    interval_range, interval_gap = compute_interval_constraints(xs, sim, options)
    if options['method'] == REDUCED:
        a = np.diag(-np.ones(num_categories), -1) + np.diag(np.ones(num_categories + 1))
        a = a[:-1, :-1]
        b = np.empty((num_categories,))
        b[0] = interval_range
        b[1:] = interval_range + interval_gap
    elif options['method'] == STANDARD:
        a = np.diag(-np.ones(2 * num_categories), -1) + np.diag(np.ones(2 * num_categories + 1))
        a = a[:-1, :]
        a = a[:, :-1]
        b = np.empty((2 * num_categories,))
        b[0] = 0
        b[1::2] = interval_range
        b[2::2] = interval_gap
    ineq_cons = {'type': 'ineq', 'fun': lambda x: a.dot(x) - b}

    return ineq_cons
