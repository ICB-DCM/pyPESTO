from typing import Callable, Literal

import numpy as np

from ..problem import Problem
from ..result import ProfilerResult
from .options import ProfileOptions

__all__ = ['next_guess', 'fixed_step', 'adaptive_step']


def next_guess(
    x: np.ndarray,
    par_index: int,
    par_direction: Literal[1, -1],
    profile_options: ProfileOptions,
    update_type: Literal[
        'fixed_step',
        'adaptive_step_order_0',
        'adaptive_step_order_1',
        'adaptive_step_regression',
    ],
    current_profile: ProfilerResult,
    problem: Problem,
    global_opt: float,
) -> np.ndarray:
    """
    Create the next initial guess for the optimizer.

    Used in order to compute the next profile point. Different proposal methods
    are available.

    Parameters
    ----------
    x:
        The current position of the profiler.
    par_index:
        The index of the parameter of the current profile.
    par_direction:
        The direction, in which the profiling is done (``1`` or ``-1``).
    profile_options:
        Various options applied to the profile optimization.
    update_type:
        Type of update for next profile point. Available options are:

        * ``fixed_step`` (see :func:`fixed_step`)
        * ``adaptive_step_order_0`` (see :func:`adaptive_step`).
        * ``adaptive_step_order_1`` (see :func:`adaptive_step`).
        * ``adaptive_step_regression`` (see :func:`adaptive_step`).
    current_profile:
        The profile which should be computed.
    problem:
        The problem to be solved.
    global_opt:
        Log-posterior value of the global optimum.

    Returns
    -------
    The next initial guess as base for the next profile point.
    """
    if update_type == 'fixed_step':
        return fixed_step(
            x, par_index, par_direction, profile_options, problem
        )

    if update_type == 'adaptive_step_order_0':
        order = 0
    elif update_type == 'adaptive_step_order_1':
        order = 1
    elif update_type == 'adaptive_step_regression':
        order = np.nan
    else:
        raise ValueError(
            f'Unsupported `update_type` {update_type} for `next_guess`.'
        )

    return adaptive_step(
        x,
        par_index,
        par_direction,
        profile_options,
        current_profile,
        problem,
        global_opt,
        order,
    )


def fixed_step(
    x: np.ndarray,
    par_index: int,
    par_direction: Literal[1, -1],
    options: ProfileOptions,
    problem: Problem,
) -> np.ndarray:
    """Most simple method to create the next guess.

    Computes the next point based on the fixed step size given by
    :attr:`pypesto.profile.ProfileOptions.default_step_size`.

    Parameters
    ----------
    x:
       The current position of the profiler, size `dim_full`.
    par_index:
        The index of the parameter of the current profile.
    par_direction:
        The direction, in which the profiling is done (``1`` or ``-1``).
    options:
        Various options applied to the profile optimization.
    problem:
        The problem to be solved.

    Returns
    -------
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
    par_direction: Literal[1, -1],
    options: ProfileOptions,
    current_profile: ProfilerResult,
    problem: Problem,
    global_opt: float,
    order: int = 1,
) -> np.ndarray:
    """Group of more complex methods for point proposal.

    Step size is automatically computed by a line search algorithm (hence:
    adaptive).

    Parameters
    ----------
    x:
        The current position of the profiler, size `dim_full`.
    par_index:
        The index of the parameter of the current profile.
    par_direction:
        The direction, in which the profiling is done (``1`` or ``-1``).
    options:
        Various options applied to the profile optimization.
    current_profile:
        The profile which should be computed.
    problem:
        The problem to be solved.
    global_opt:
        Log-posterior value of the global optimum.
    order:
        Specifies the precise algorithm for extrapolation.
        Available options are:

        * ``0``: just one parameter is updated
        * ``1``: the last two points are used to extrapolate all parameters
        * ``np.nan``: indicates that a more complex regression should be used
          as determined by :attr:`pypesto.profile.ProfileOptions.reg_order`.


    Returns
    -------
    The updated parameter vector, of size `dim_full`.
    """

    # restrict step proposal to minimum and maximum step size
    def clip_to_minmax(step_size_proposal):
        return np.clip(
            step_size_proposal, options.min_step_size, options.max_step_size
        )

    # restrict step proposal to bounds
    def clip_to_bounds(step_proposal):
        return np.clip(step_proposal, problem.lb_full, problem.ub_full)

    problem.fix_parameters(par_index, x[par_index])

    # Get update directions and first step size guesses
    (
        step_size_guess,
        delta_x_dir,
        reg_par,
        delta_obj_value,
    ) = handle_profile_history(
        x,
        par_index,
        par_direction,
        global_opt,
        order,
        current_profile,
        problem,
        options,
    )

    # check whether we must make a minimum step anyway, since we're close to
    # the next bound
    min_delta_x = x[par_index] + par_direction * options.min_step_size

    if par_direction == -1 and (min_delta_x < problem.lb_full[par_index]):
        step_length = problem.lb_full[par_index] - x[par_index]
        return x + step_length * delta_x_dir

    if par_direction == 1 and (min_delta_x > problem.ub_full[par_index]):
        step_length = problem.ub_full[par_index] - x[par_index]
        return x + step_length * delta_x_dir

    # parameter extrapolation function
    n_profile_points = len(current_profile.fval_path)

    # Do we have enough points to do a regression?
    if np.isnan(order) and n_profile_points > 2:

        def par_extrapol(step_length):
            x_step = []
            # loop over parameters, extrapolate each one
            for i_par in range(problem.dim_full):
                if i_par == par_index:
                    # if we meet the profiling parameter, just increase,
                    # don't extrapolate
                    x_step.append(x[par_index] + step_length * par_direction)
                elif i_par in problem.x_fixed_indices:
                    # common fixed parameter: will be ignored anyway later
                    x_step.append(np.nan)
                else:
                    # extrapolate
                    cur_par_extrapol = np.poly1d(reg_par[i_par])
                    x_step.append(
                        cur_par_extrapol(
                            x[par_index] + step_length * par_direction
                        )
                    )
            return clip_to_bounds(x_step)

    else:
        # if not, we do simple extrapolation
        def par_extrapol(step_length):
            x_step = x + step_length * delta_x_dir
            return clip_to_bounds(x_step)

    # compute proposal
    next_x = par_extrapol(step_size_guess)

    # next start point has to be searched
    # compute the next objective value which we aim for
    next_obj_target = (
        -np.log(1.0 - options.delta_ratio_max)
        + options.magic_factor_obj_value * delta_obj_value
        + current_profile.fval_path[-1]
    )

    # compute objective at the guessed point
    problem.fix_parameters(par_index, next_x[par_index])
    next_obj = problem.objective(problem.get_reduced_vector(next_x))

    # iterate until good step size is found
    return do_line_search(
        next_x,
        step_size_guess,
        par_extrapol,
        next_obj,
        next_obj_target,
        clip_to_minmax,
        clip_to_bounds,
        par_index,
        problem,
        options,
    )


def handle_profile_history(
    x: np.ndarray,
    par_index: int,
    par_direction: Literal[1, -1],
    global_opt: float,
    order: int,
    current_profile: ProfilerResult,
    problem: Problem,
    options: ProfileOptions,
) -> tuple[float, np.array, list[float], float]:
    """Compute the very first step direction update guesses.

    Check whether enough steps have been taken for applying regression,
    computes regression or simple extrapolation.

    Returns
    -------
    step_size_guess:
        Guess for the step size.
    delta_x_dir:
        Parameter update direction.
    reg_par:
        The regression polynomial for profile extrapolation.
    delta_obj_value:
        The difference of the objective function value between the last point and `global_opt`.
    """
    n_profile_points = len(current_profile.fval_path)

    # set the update direction
    delta_x_dir = np.zeros(len(x))
    delta_x_dir[par_index] = par_direction
    reg_par = None

    # Is this the first step along this profile? If so, try a simple step
    if n_profile_points == 1:
        # try to use the default step size
        step_size_guess = options.default_step_size
        delta_obj_value = 0.0

    else:
        # try to reuse the previous step size
        step_size_guess = np.abs(
            current_profile.x_path[par_index, -1]
            - current_profile.x_path[par_index, -2]
        )
        delta_obj_value = current_profile.fval_path[-1] - global_opt

        if order == 1 or (np.isnan(order) and n_profile_points < 3):
            # set the update direction (extrapolate with order 1)
            last_delta_x = (
                current_profile.x_path[:, -1] - current_profile.x_path[:, -2]
            )
            delta_x_dir = last_delta_x / step_size_guess
        elif np.isnan(order):
            # compute the regression polynomial for parameter extrapolation
            reg_par = get_reg_polynomial(
                par_index, current_profile, problem, options
            )

    return step_size_guess, delta_x_dir, reg_par, delta_obj_value


def get_reg_polynomial(
    par_index: int,
    current_profile: ProfilerResult,
    problem: Problem,
    options: ProfileOptions,
) -> list[float]:
    """Compute the regression polynomial.

    Used to step proposal extrapolation from the last profile points.
    """
    # determine interpolation order
    n_profile_points = len(current_profile.fval_path)
    reg_max_order = np.floor(n_profile_points / 2)
    reg_order = min(reg_max_order, options.reg_order)
    reg_points = min(n_profile_points, options.reg_points)

    # set up matrix of regression parameters
    reg_par = []
    for i_par in range(problem.dim_full):
        if i_par in problem.x_fixed_indices:
            # if we meet the current profiling parameter or a fixed parameter,
            # there is nothing to do, so pass a np.nan
            reg_par.append(np.nan)
        else:
            # Do polynomial interpolation of profile path
            # Determine rank of polynomial interpolation
            regression_tmp = np.polyfit(
                current_profile.x_path[par_index, -reg_points:],
                current_profile.x_path[i_par, -reg_points:],
                reg_order,
                full=True,
            )

            # Decrease rank if interpolation problem is ill-conditioned
            if regression_tmp[2] < reg_order:
                reg_order = regression_tmp[2]
                regression_tmp = np.polyfit(
                    current_profile.x_path[par_index, -reg_points:],
                    current_profile.x_path[i_par, -reg_points:],
                    int(reg_order),
                    full=True,
                )

            # add to regression parameters
            reg_par.append(regression_tmp[0])

    return reg_par


def do_line_search(
    next_x: np.ndarray,
    step_size_guess: float,
    par_extrapol: Callable,
    next_obj: float,
    next_obj_target: float,
    clip_to_minmax: Callable,
    clip_to_bounds: Callable,
    par_index: int,
    problem: Problem,
    options: ProfileOptions,
) -> np.ndarray:
    """Perform the line search.

    Based on the objective function we want to reach, based on the current
    position in parameter space and on the first guess for the proposal.

    Parameters
    ----------
    next_x:
        Starting parameters for the line search.
    step_size_guess:
        First guess for the step size.
    par_extrapol:
        Parameter extrapolation function.
    next_obj:
        Objective function value at `next_x`.
    next_obj_target:
        Objective function value we want to reach.
    clip_to_minmax:
        Function to clip the step size to minimum and maximum step size.
    clip_to_bounds:
        Function to clip the parameters to the bounds.
    par_index:
        Index of the parameter we are profiling.
    problem:
        The parameter estimation problem.
    options:
        Profile likelihood options.

    Returns
    -------
    Parameter vector that is expected to yield the objective function value
    closest to `next_obj_target`.
    """
    # Was the initial step too big or too small?
    direction = "decrease" if next_obj_target < next_obj else "increase"
    if direction == 'increase':
        adapt_factor = options.step_size_factor
    else:
        adapt_factor = 1 / options.step_size_factor

    # Loop until correct step size was found
    while True:
        # Adapt step size of guess
        last_x = next_x
        step_size_guess = clip_to_minmax(step_size_guess * adapt_factor)
        next_x = clip_to_bounds(par_extrapol(step_size_guess))

        # Check if we hit the bounds
        if (
            direction == 'decrease'
            and step_size_guess == options.min_step_size
        ):
            return next_x
        if (
            direction == 'increase'
            and step_size_guess == options.max_step_size
        ):
            return next_x

        # compute new objective value
        problem.fix_parameters(par_index, next_x[par_index])
        last_obj = next_obj
        next_obj = problem.objective(problem.get_reduced_vector(next_x))

        # check for root crossing and compute correct step size in case
        if (direction == 'decrease' and next_obj_target >= next_obj) or (
            direction == 'increase' and next_obj_target <= next_obj
        ):
            return next_x_interpolate(
                next_obj, last_obj, next_x, last_x, next_obj_target
            )


def next_x_interpolate(
    next_obj: float,
    last_obj: float,
    next_x: np.ndarray,
    last_x: np.ndarray,
    next_obj_target: float,
) -> np.ndarray:
    """Interpolate between the last two steps."""
    delta_obj = np.abs(next_obj - last_obj)
    add_x = np.abs(last_obj - next_obj_target) * (next_x - last_x) / delta_obj

    # fix final guess and return
    return last_x + add_x
