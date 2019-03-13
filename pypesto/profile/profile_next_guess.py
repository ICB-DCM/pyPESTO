import numpy as np
import copy


def next_guess(x,
               par_index,
               par_direction,
               profile_options,
               update_type,
               current_profile,
               problem,
               global_opt):
    """
    This function creates the next inital guess for the optimizer in
    order to compute the next profile point. Different proposal methods
    are available.

    Parameters
    ----------

    x: numpy.ndarray
       The current position of the profiler

    par_index: int
        The index of the parameter of the current profile

    par_direction: int
        The direction, in which the profiling is done (1 or -1)

    profile_options: pypesto.ProfileOptions, optional
        Various options applied to the profile optimization.

    update_type: str
        type of update for next profile point

    current_profile: pypesto.ProfilerResults
        The profile which should be computed

    problem: pypesto.Problem
        The problem to be solved.

    global_opt: float
        log-posterior value of the global optimum
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


def fixed_step(x,
               par_index,
               par_direction,
               options,
               problem):
    """
       Most simple method to create the next guess.

       Parameters
       ----------

        x: ndarray
           The current position of the profiler

        par_index: int
            The index of the parameter of the current profile

        par_direction: int
            The direction, in which the profiling is done (1 or -1)

        options: pypesto.ProfileOptions, optional
            Various options applied to the profile optimization.

        problem: pypesto.Problem
            The problem to be solved.
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


def adaptive_step(x, par_index, par_direction, options, current_profile,
                  problem, global_opt, order=1):
    """
       group of more complex methods for point proposal, step size is
       automatically computed by a line search algorithm (hence: adaptive)

       Parameters
       ----------

        x: ndarray
           The current position of the profiler

        par_index: int
            The index of the parameter of the current profile

        par_direction: int
            The direction, in which the profiling is done (1 or -1)

        options: pypesto.ProfileOptions
            Various options applied to the profile optimization.

        current_profile: pypesto.ProfilerResults
            The profile which should be computed

        problem: pypesto.Problem
            The problem to be solved.

        global_opt: float
            log-posterior value of the global optimum

        order: int
            specifies the precise algorithm for extrapolation: can be 0 (
            just one parameter is updated), 1 (last two points used to
            extrapolate all parameters), and np.nan (indicates that a more
            complex regression should be used)
       """

    # restrict step proposal to minimum and maximum step size
    def clip_to_minmax(step_size_proposal):
        return clip_vector(step_size_proposal, options.min_step_size,
                           options.max_step_size)

    # restrict step proposal to bounds
    def clip_to_bounds(step_proposal):
        return clip_vector(step_proposal, problem.lb_full, problem.ub_full)

    # check if this is the first step
    n_profile_points = len(current_profile.fval_path)
    pos_ind_red = len([ip for ip in problem.x_free_indices if ip < par_index])
    problem.fix_parameters(par_index, x[par_index])

    # Get update directions and first step size guesses
    (step_size_guess, delta_x_dir, reg_par, delta_obj_value) = \
        handle_profile_history(x, par_index, par_direction, n_profile_points,
                               pos_ind_red, global_opt, order,
                               current_profile, problem, options)

    # check whether we must make a minimum step anyway, since we're close to
    # the next bound
    min_delta_x = x[par_index] + par_direction * options.min_step_size
    if par_direction is -1:
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
            for i_par in range(len(problem.x_free_indices) + 1):
                if i_par == pos_ind_red:
                    # if we meet the profiling parameter, just increase,
                    # don't extrapolate
                    x_step_tmp.append(x[par_index] + step_length *
                                      par_direction)
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
    next_obj = problem.objective(reduce_x(next_x, par_index))

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


def handle_profile_history(x, par_index, par_direction, n_profile_points,
                           pos_ind_red, global_opt, order, current_profile,
                           problem, options):
    """
       Computes the very first step direction update guesses, check whether
       enough step have been taken for applying regression, computes
       regression or simple extrapolation
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

            reg_par = get_reg_polynomial(n_profile_points, pos_ind_red,
                                         par_index, current_profile,
                                         problem, options)

    return step_size_guess, delta_x_dir, reg_par, delta_obj_value


def get_reg_polynomial(n_profile_points, pos_ind_red, par_index,
                       current_profile, problem, options):
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
    for i_par in range(len(problem.x_free_indices) + 1):
        if i_par == pos_ind_red:
            # if we meet the current profiling parameter, there is nothing
            # to do, so pass an np.nan
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
                    reg_order.astype(int), full=True)

            # add to regression parameters
            reg_par.append(regression_tmp[0])

    return reg_par


def do_line_seach(next_x, step_size_guess, direction, par_extrapol, next_obj,
                  next_obj_target, clip_to_minmax, clip_to_bounds,
                  par_index, problem, options):
    """
       Performs the line search based on the objective function we want to
       reach, based on the current position in parameter space and on the
       first guess for the proposal
    """

    # Was the inital step too big or too small?
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
                      options.min_step_size) or (direction == 'increase' and
                                                 step_size_guess ==
                                                 options.max_step_size)

        if hit_bounds:
            return next_x
        else:
            # compute new objective value
            problem.fix_parameters(par_index, next_x[par_index])
            last_obj = copy.copy(next_obj)
            next_obj = problem.objective(reduce_x(next_x, par_index))

            # check for root crossing and compute correct step size in case
            if direction == 'decrease' and next_obj_target >= next_obj:
                return next_x_interpolate(next_obj, last_obj, next_x,
                                          last_x, next_obj_target)
            elif direction == 'increase' and next_obj_target <= next_obj:
                return next_x_interpolate(next_obj, last_obj, next_x,
                                          last_x, next_obj_target)


def next_x_interpolate(next_obj, last_obj, next_x, last_x, next_obj_target):
    """
       interpolate between the last two steps
    """
    delta_obj = np.abs(next_obj - last_obj)
    add_x = np.abs(last_obj - next_obj_target) * (
            next_x - last_x) / delta_obj

    # fix final guess and return
    return last_x + add_x


def reduce_x(next_x, par_index):
    """
       reduce step proposal to non-fixed parameters
    """
    red_ind = list(range(len(next_x)))
    red_ind.pop(par_index)
    return np.array([next_x[ip] for ip in red_ind])


def clip_vector(vector_guess, lower, upper):
    """
       restrict a scalar or a vector to given bounds
    """
    if isinstance(vector_guess, float):
        vector_guess = np.max([np.min([vector_guess, upper]), lower])
    else:
        for i_par, i_guess in enumerate(vector_guess):
            vector_guess[i_par] = np.max([np.min([i_guess, upper[i_par]]),
                                          lower[i_par]])
    return vector_guess
