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
    if par_direction is -1 and next_x_par < problem.lb_full[par_index]:
        delta_x[par_index] = problem.lb_full[par_index] - x[par_index]
    elif par_direction is 1 and next_x_par > problem.ub_full[par_index]:
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

    # There is this magic factor in the old profiling code which slows down
    # profiling at small ratios (must be >= 0 and < 1)
    magic_factor_obj_value = options.magic_factor_obj_value

    # check if this is the first step, compute the direction for the first
    # guess of next step
    n_profile_points = len(current_profile.fval_path)
    pos_ind_red = len([ip for ip in problem.x_free_indices if ip < par_index])
    problem.fix_parameters(par_index, x[par_index])

    if n_profile_points == 1:
        # take default step size
        step_size_guess = options.default_step_size
        delta_obj_value = 0.

        # set the update direction
        delta_x_dir = np.zeros(len(x))
        delta_x_dir[par_index] = par_direction
    else:
        # take last step size
        step_size_guess = np.abs(current_profile.x_path[par_index, -1] -
                                 current_profile.x_path[par_index, -2])
        delta_obj_value = current_profile.fval_path[-1] - global_opt

        if order == 0:
            # set the update direction
            delta_x_dir = np.zeros(len(x))
            delta_x_dir[par_index] = par_direction
        elif order == 1 or (np.isnan(order) and n_profile_points < 3):
            # set the update direction
            last_delta_x = current_profile.x_path[:, -1] - \
                           current_profile.x_path[:, -2]
            step_size_guess = np.abs(current_profile.x_path[par_index, -1] -
                                     current_profile.x_path[par_index, -2])
            delta_x_dir = last_delta_x / step_size_guess
        elif np.isnan(order):
            # disable rank warnings
            # warnings.simplefilter('ignore', np.RankWarning)

            # determine interpolation order
            reg_max_order = np.floor(n_profile_points / 2)
            reg_order = np.min([reg_max_order, options.reg_order])
            reg_points = np.min([n_profile_points, options.reg_points])
            reg_par = []
            for i_par in range(len(problem.x_free_indices) + 1):
                if i_par == pos_ind_red:
                    reg_par.append(np.nan)
                else:
                    # Do polynomial interpolation of profile path
                    # Determine rank of polynomial interpolation
                    regression_tmp = np.polyfit(
                        current_profile.x_path[par_index, -1:-reg_points:-1],
                        current_profile.x_path[i_par, -1:-reg_points:-1],
                        reg_order, full=True)

                    # Decrease rank if interpolation problem is illconditioned
                    if regression_tmp[2] < reg_order:
                        reg_order = regression_tmp[2]
                        regression_tmp = np.polyfit(
                            current_profile.x_path[par_index, -reg_points:-1],
                            current_profile.x_path[i_par, -reg_points:-1],
                            reg_order.astype('int'), full=True)

                    reg_par.append(regression_tmp[0])

    # boolean indicating whether a search should be carried out
    search = True

    # check whether we must make a minimum step anyway, since we're close to
    # the next bound
    min_delta_x = x[par_index] + par_direction * options.min_step_size
    if par_direction is -1:
        if min_delta_x < problem.lb_full[par_index]:
            step_length = problem.lb_full[par_index] - x[par_index]
            search = False
    else:
        if min_delta_x > problem.ub_full[par_index]:
            step_length = problem.ub_full[par_index] - x[par_index]
            search = False

    if not search:
        return x + step_length * delta_x_dir

    # restrict step proposal to minimum and maximum step size
    def clip_to_minmax(step_size_proposal):
        return clip_vector(step_size_proposal, options.min_step_size,
                           options.max_step_size)

    # restrict step proposal to bounds
    def clip_to_bounds(step_proposal):
        return clip_vector(step_proposal, problem.lb_full, problem.ub_full)

    # parameter extrapolation function
    def par_extrapol(step_length):
        if np.isnan(order) and n_profile_points > 2:
            # if we do regression
            x_step_tmp = []
            for i_par in range(len(problem.x_free_indices) + 1):
                if i_par == pos_ind_red:
                    x_step_tmp.append(x[par_index] + step_length *
                                      par_direction)
                else:
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
        magic_factor_obj_value * delta_obj_value + \
        current_profile.fval_path[-1]

    # compute objective at the guessed point
    problem.fix_parameters(par_index, next_x[par_index])
    next_obj = problem.objective(reduce_x(next_x, par_index))


    # iterate until good step size is found
    if next_obj_target < next_obj:
        # The step is rather too long
        stop_search = False
        while not stop_search:
            # Reduce step size of guess
            last_x = copy.copy(next_x)
            step_size_guess = clip_to_minmax(step_size_guess /
                                             options.step_size_factor)
            next_x = clip_to_bounds(par_extrapol(step_size_guess))

            # Check if we crossed a root or reduced to the minimum
            if step_size_guess == options.min_step_size:
                stop_search = True
            else:
                # compute new objective value
                problem.fix_parameters(par_index, next_x[par_index])
                last_obj = copy.copy(next_obj)
                next_obj = problem.objective(reduce_x(next_x, par_index))

                # check for root crossing
                if next_obj_target >= next_obj:
                    stop_search = True

                    # interpolate between the last two steps
                    delta_obj = np.abs(next_obj - last_obj)
                    add_x = np.abs(last_obj - next_obj_target) * (
                            next_x - last_x) / delta_obj

                    # fix final guess
                    next_x = last_x + add_x

    else:
        # The step is rather too short
        stop_search = False
        while not stop_search:
            last_x = copy.copy(next_x)
            step_size_guess = clip_to_minmax(step_size_guess *
                                             options.step_size_factor)
            next_x = clip_to_bounds(par_extrapol(step_size_guess))

            # Check if we crossed a root or increased to the maximum
            if step_size_guess == options.max_step_size:
                stop_search = True
            else:
                # compute new objective value
                problem.fix_parameters(par_index, next_x[par_index])
                last_obj = copy.copy(next_obj)
                next_obj = problem.objective(reduce_x(next_x, par_index))

                # check for root crossing
                if next_obj_target <= next_obj:
                    stop_search = True

                    # interpolate between the last two steps
                    delta_obj = np.abs(next_obj - last_obj)
                    add_x = np.abs(last_obj - next_obj_target) * (
                            next_x - last_x) / delta_obj

                    # fix final guess
                    next_x = last_x + add_x

    return next_x

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
