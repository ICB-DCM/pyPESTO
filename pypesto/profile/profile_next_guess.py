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
       This is function initializes profiling based on a previous optimization.

       Parameters
       ----------

        x: ndarray
           The current position of the profiler

        par_index: ndarray
            The index of the current profile

        par_direction: int
            The direction, in which the profiling is done (1 or -1)

        profile_options: pypesto.ProfileOptions, optional
            Various options applied to the profile optimization.

        update_type: basestring
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
    else:
        if update_type == 'adaptive_step_order_0':
            order = 0
        elif update_type == 'adaptive_step_order_1':
            order = 1

        return adaptive_step(x, par_index, par_direction, profile_options,
                             current_profile, problem, global_opt, order)


def fixed_step(x, par_index, par_direction, options, problem):
    delta_x = np.zeros(len(x))
    delta_x[par_index] = par_direction * options.default_step_size

    # check whether the next point is maybe outside the bounds
    # and correct it
    next_x_par = x[par_index] + delta_x[par_index]
    if par_direction is -1:
        if next_x_par < problem.lb_full[par_index]:
            delta_x[par_index] = problem.lb_full[par_index] - x[par_index]
    else:
        if next_x_par > problem.ub_full[par_index]:
            delta_x[par_index] = problem.ub_full[par_index] - x[par_index]

    return x + delta_x


def adaptive_step(x, par_index, par_direction, options, current_profile,
                  problem, global_opt, order=1):
    # There is this magic factor in the old profiling code which slows down
    # profiling at small ratios (must be >= 0 and < 1)
    magic_factor_obj_value = 0.25

    # check if this is the first step, compute the direction for the first
    # guess of next step
    n_profile_points = len(current_profile.fval_path)
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
        elif order == 1:
            # set the update direction
            last_delta_x = current_profile.x_path[:, -1] - \
                           current_profile.x_path[:, -2]
            step_size_guess = np.abs(current_profile.x_path[par_index, -1] -
                                     current_profile.x_path[par_index, -2])
            delta_x_dir = last_delta_x / step_size_guess
        elif np.isnan(order):
            reg_max_order = np.floor(n_profile_points / 2)
            reg_order = np.min(reg_max_order, options.reg_order)
            reg_points = np.min(n_profile_points, options.reg_points)
            reg_par = []
            for i_par in problem.x_free_indices:
                reg_par.append(np.polyfit(
                    current_profile.x_path[par_index, -reg_points:-1],
                    current_profile.x_path[i_par, -reg_points:-1], reg_order))

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

    # restrict a step to the bounds
    def clip_to_bounds(step_proposal):
        for i_par, i_step in enumerate(step_proposal):
            step_proposal[i_par] = np.max([np.min([i_step, problem.ub_full[
                i_par]]), problem.lb_full[i_par]])
        return step_proposal

    # parameter extrapolation function
    def par_extrapol(step_length):
        x_step = x + step_length * delta_x_dir
        return clip_to_bounds(x_step)

    # parameter reduction function (cutting out the entry par_index)
    def reduce_x(next_x):
        red_ind = list(range(0, len(next_x)))
        red_ind.pop(par_index)
        return np.array([next_x[ip] for ip in red_ind])

    # compute proposal
    next_x = par_extrapol(step_size_guess)

    # next start point has to be searched
    # compute the next objective value which we aim for
    next_obj_target = - np.log(1. - options.delta_ratio_max) - \
                      magic_factor_obj_value * delta_obj_value + \
                      current_profile.fval_path[-1]

    # compute objective at the guessed point
    problem.fix_parameters(par_index, next_x[par_index])
    next_obj = problem.objective(reduce_x(next_x))

    # restrict a step size to min and max
    def clip_to_minmax(step_size_proposal):
        step_size_proposal = np.max(
            [np.min([step_size_proposal, options.max_step_size]),
             options.min_step_size])
        return step_size_proposal

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
                next_obj = problem.objective(reduce_x(next_x))

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
                next_obj = problem.objective(reduce_x(next_x))

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
