import numpy as np
import copy
from typing import Dict, List

from ..objective import Objective
from ..problem import Problem
from ..optimize import minimize, Optimizer
from .parameter import InnerParameter
from .problem import InnerProblem, scale_value_dict
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
