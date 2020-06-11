import numpy as np
import copy
from typing import Callable, List, Tuple, Union

from ..problem import Problem
from .options import ProfileOptions
from .result import ProfilerResult


def next_guess(
        x: np.ndarray,
        par_index: int,
        par_direction: int,
        profile_options: ProfileOptions,
        update_type: str,
        current_profile: ProfilerResult,
        problem: Problem,
        global_opt: float
) -> np.ndarray:
    """
    This function creates the next initial guess for the optimizer in
    order to compute the next profile point. Different proposal methods
    are available.

    Parameters
    ----------
    x:
        The current position of the profiler.
    par_index:
        The index of the parameter of the current profile.
    par_direction:
        The direction, in which the profiling is done (1 or -1).
    profile_options:
        Various options applied to the profile optimization.
    update_type:
        Type of update for next profile point.
    current_profile:
        The profile which should be computed.
    problem:
        The problem to be solved.
    global_opt:
        Log-posterior value of the global optimum.

    Returns
    -------
    next_guess:
        The next initial guess as base for the next profile point.
    """

    if update_type == 'fixed_step':
        return fixed_step(x, par_index, par_direction, profile_options,
                          problem)
    elif update_type == 'adaptive_step_order_0':
        order = 0
    elif update_type == 'adaptive_step_order_1':
        order = 1
    elif update_type == 'adaptive_step_regression':
        order = np.nan
    else:
        raise Exception('Unsupported update_type for '
                        'create_next_startpoint.')

    return adaptive_step(x, par_index, par_direction, profile_options,
                         current_profile, problem, global_opt, order)


def fixed_step(
        x: np.ndarray,
        par_index: int,
        par_direction: int,
        options: ProfileOptions,
        problem: Problem
) -> np.ndarray:
    """
    Most simple method to create the next guess.

    Parameters
    ----------
    x:
       The current position of the profiler, size `dim_full`.
    par_index:
        The index of the parameter of the current profile
    par_direction:
        The direction, in which the profiling is done (1 or -1)
    options:
        Various options applied to the profile optimization.
    problem:
        The problem to be solved.

    Returns
    -------
    x_new:
        The updated parameter vector, of size `dim_full`.
    """
    delta_x = np.zeros(len(x))
    delta_x[par_index] = par_direction * options.default_step_size

    # check whether the next point is maybe outside the bounds
    # and correct it
    next_x_par = x[par_index] + delta_x[par_index]
    if par_direction == -1 and next_x_par < problem.lb_full[par_index]:
        delta_x[par_index] = problem.lb_full[par_index] - x[par_index]
    elif par_direction == 1 and next_x_par > problem.ub_full[par_index]:
        delta_x[par_index] = problem.ub_full[par_index] - x[par_index]

    return x + delta_x


def adaptive_step(
        x: np.ndarray,
        par_index: int,
        par_direction: int,
        options: ProfileOptions,
        current_profile: ProfilerResult,
        problem: Problem,
        global_opt: float,
        order: int = 1
) -> np.ndarray:
    """
    group of more complex methods for point proposal, step size is
    automatically computed by a line search algorithm (hence: adaptive)

    Parameters
    ----------
    x:
        The current position of the profiler, size `dim_full`.
    par_index:
        The index of the parameter of the current profile
    par_direction:
        The direction, in which the profiling is done (1 or -1)
    options:
        Various options applied to the profile optimization.
    current_profile:
        The profile which should be computed
    problem:
        The problem to be solved.
    global_opt:
        log-posterior value of the global optimum
    order:
        Specifies the precise algorithm for extrapolation: can be 0 (
        just one parameter is updated), 1 (last two points used to
        extrapolate all parameters), and np.nan (indicates that a more
        complex regression should be used)

    Returns
    -------
    x_new:
        The updated parameter vector, of size `dim_full`.
    """
    # restrict step proposal to minimum and maximum step size
    def clip_to_minmax(step_size_proposal):
        return clip(step_size_proposal, options.min_step_size,
                    options.max_step_size)

    # restrict step proposal to bounds
    def clip_to_bounds(step_proposal):
        return clip(step_proposal, problem.lb_full, problem.ub_full)

    # check if this is the first step
    n_profile_points = len(current_profile.fval_path)
    problem.fix_parameters(par_index, x[par_index])

    # Get update directions and first step size guesses
    (step_size_guess, delta_x_dir, reg_par, delta_obj_value) = \
        handle_profile_history(x, par_index, par_direction, n_profile_points,
                               global_opt, order,
                               current_profile, problem, options)

    # check whether we must make a minimum step anyway, since we're close to
    # the next bound
    min_delta_x = x[par_index] + par_direction * options.min_step_size
    if par_direction == -1:
        if min_delta_x < problem.lb_full[par_index]:
            step_length = problem.lb_full[par_index] - x[par_index]
            return x + step_length * delta_x_dir
    else:
        if min_delta_x > problem.ub_full[par_index]:
            step_length = problem.ub_full[par_index] - x[par_index]
            return x + step_length * delta_x_dir

    # parameter extrapolation function
    def par_extrapol(step_length):
        # Do we have enough points to do a regression?
        if np.isnan(order) and n_profile_points > 2:
            x_step_tmp = []
            # loop over parameters, extrapolate each one
            for i_par in range(problem.dim_full):
                if i_par == par_index:
                    # if we meet the profiling parameter, just increase,
                    # don't extrapolate
                    x_step_tmp.append(x[par_index] + step_length *
                                      par_direction)
                elif i_par in problem.x_fixed_indices:
                    # common fixed parameter: will be ignored anyway later
                    x_step_tmp.append(np.nan)
                else:
                    # extrapolate
                    cur_par_extrapol = np.poly1d(reg_par[i_par])
                    x_step_tmp.append(cur_par_extrapol(x[par_index] +
                                                       step_length *
                                                       par_direction))
            x_step = np.array(x_step_tmp)
        else:
            # if we do simple extrapolation
            x_step = x + step_length * delta_x_dir

        return clip_to_bounds(x_step)

    # compute proposal
    next_x = par_extrapol(step_size_guess)

    # next start point has to be searched
    # compute the next objective value which we aim for
    next_obj_target = - np.log(1. - options.delta_ratio_max) + \
        options.magic_factor_obj_value * delta_obj_value + \
        current_profile.fval_path[-1]

    # compute objective at the guessed point
    problem.fix_parameters(par_index, next_x[par_index])
    next_obj = problem.objective(problem.get_reduced_vector(next_x))

    # iterate until good step size is found
    if next_obj_target < next_obj:
        # The step is rather too long
        return do_line_seach(next_x, step_size_guess, 'decrease',
                             par_extrapol, next_obj, next_obj_target,
                             clip_to_minmax, clip_to_bounds, par_index,
                             problem, options)

    else:
        # The step is rather too short
        return do_line_seach(next_x, step_size_guess, 'increase',
                             par_extrapol, next_obj, next_obj_target,
                             clip_to_minmax, clip_to_bounds, par_index,
                             problem, options)


def handle_profile_history(
        x: np.ndarray,
        par_index: int,
        par_direction: int,
        n_profile_points: int,
        global_opt: float,
        order: int,
        current_profile: ProfilerResult,
        problem: Problem,
        options: ProfileOptions
) -> Tuple:
    """
    Computes the very first step direction update guesses, check whether
    enough steps have been taken for applying regression, computes
    regression or simple extrapolation.
    """

    # set the update direction
    delta_x_dir = np.zeros(len(x))
    delta_x_dir[par_index] = par_direction
    reg_par = None

    # Is this the first step along this profile? If so, try a simple step
    if n_profile_points == 1:
        # try to use the default step size
        step_size_guess = options.default_step_size
        delta_obj_value = 0.

    else:
        # try to reuse the previous step size
        step_size_guess = np.abs(current_profile.x_path[par_index, -1] -
                                 current_profile.x_path[par_index, -2])
        delta_obj_value = current_profile.fval_path[-1] - global_opt

        if order == 1 or (np.isnan(order) and n_profile_points < 3):
            # set the update direction (extrapolate with order 1)
            last_delta_x = current_profile.x_path[:, -1] - \
                           current_profile.x_path[:, -2]
            step_size_guess = np.abs(current_profile.x_path[par_index, -1] -
                                     current_profile.x_path[par_index, -2])
            delta_x_dir = last_delta_x / step_size_guess
        elif np.isnan(order):
            # compute the regression polynomial for parameter extrapolation

            reg_par = get_reg_polynomial(n_profile_points,
                                         par_index, current_profile,
                                         problem, options)

    return step_size_guess, delta_x_dir, reg_par, delta_obj_value


def get_reg_polynomial(
        n_profile_points: int,
        par_index: int,
        current_profile: ProfilerResult,
        problem: Problem,
        options: ProfileOptions
) -> List[float]:
    """
    Computes the regression polynomial which is used to step proposal
    extrapolation from the last profile points
    """

    # determine interpolation order
    reg_max_order = np.floor(n_profile_points / 2)
    reg_order = np.min([reg_max_order, options.reg_order])
    reg_points = np.min([n_profile_points, options.reg_points])

    # set up matrix of regression parameters
    reg_par = []
    for i_par in range(problem.dim_full):
        if i_par in problem.x_fixed_indices:
            # if we meet the current profiling parameter or a fixed parameter,
            # there is nothing to do, so pass an np.nan
            reg_par.append(np.nan)
        else:
            # Do polynomial interpolation of profile path
            # Determine rank of polynomial interpolation
            regression_tmp = np.polyfit(
                current_profile.x_path[par_index, -1:-reg_points:-1],
                current_profile.x_path[i_par, -1:-reg_points:-1],
                reg_order, full=True)

            # Decrease rank if interpolation problem is ill-conditioned
            if regression_tmp[2] < reg_order:
                reg_order = regression_tmp[2]
                regression_tmp = np.polyfit(
                    current_profile.x_path[par_index, -reg_points:-1],
                    current_profile.x_path[i_par, -reg_points:-1],
                    int(reg_order), full=True)

            # add to regression parameters
            reg_par.append(regression_tmp[0])

    return reg_par


def do_line_seach(
        next_x: np.ndarray,
        step_size_guess: float,
        direction: str,
        par_extrapol: Callable,
        next_obj: float,
        next_obj_target: float,
        clip_to_minmax: Callable,
        clip_to_bounds: Callable,
        par_index: int,
        problem: Problem,
        options: ProfileOptions
) -> np.ndarray:
    """
    Performs the line search based on the objective function we want to
    reach, based on the current position in parameter space and on the
    first guess for the proposal.
    """
    # Was the initial step too big or too small?
    if direction == 'increase':
        adapt_factor = options.step_size_factor
    else:
        adapt_factor = 1 / options.step_size_factor

    # Loop until correct step size was found
    stop_search = False
    while not stop_search:
        # Adapt step size of guess
        last_x = copy.copy(next_x)
        step_size_guess = clip_to_minmax(step_size_guess * adapt_factor)
        next_x = clip_to_bounds(par_extrapol(step_size_guess))

        # Check if we hit the bounds
        hit_bounds = (direction == 'decrease' and step_size_guess ==
                      options.min_step_size) or (
                direction == 'increase' and step_size_guess ==
                options.max_step_size)

        if hit_bounds:
            return next_x
        else:
            # compute new objective value
            problem.fix_parameters(par_index, next_x[par_index])
            last_obj = copy.copy(next_obj)
            next_obj = problem.objective(problem.get_reduced_vector(next_x))

            # check for root crossing and compute correct step size in case
            if direction == 'decrease' and next_obj_target >= next_obj:
                return next_x_interpolate(next_obj, last_obj, next_x,
                                          last_x, next_obj_target)
            elif direction == 'increase' and next_obj_target <= next_obj:
                return next_x_interpolate(next_obj, last_obj, next_x,
                                          last_x, next_obj_target)


def next_x_interpolate(
        next_obj: float,
        last_obj: float,
        next_x: np.ndarray,
        last_x: np.ndarray,
        next_obj_target: float
) -> np.ndarray:
    """
    Interpolate between the last two steps-
    """
    delta_obj = np.abs(next_obj - last_obj)
    add_x = np.abs(last_obj - next_obj_target) * (
            next_x - last_x) / delta_obj

    # fix final guess and return
    return last_x + add_x


def clip(
        vector_guess: Union[float, np.ndarray],
        lower: Union[float, np.ndarray],
        upper: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Restrict a scalar or a vector to given bounds.
    """
    if isinstance(vector_guess, float):
        vector_guess = np.max([np.min([vector_guess, upper]), lower])
    else:
        for i_par, i_guess in enumerate(vector_guess):
            vector_guess[i_par] = np.max([np.min([i_guess, upper[i_par]]),
                                          lower[i_par]])
    return vector_guess
