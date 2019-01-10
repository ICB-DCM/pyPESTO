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
    elif update_type == 'adaptive_step_order_0':
        return adaptive_step_order_0(x, par_index, par_direction,
                                     profile_options, current_profile,
                                     problem, global_opt)


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

    return (np.nan, x + delta_x)


def adaptive_step_order_0(x, par_index, par_direction, options,
                          current_profile, problem, global_opt):
    # There is this magic factor in the old profiling code which slows down
    # profiling at small ratios (must be >= 0 and < 1)
    magic_factor_obj_value = 0.

    # set the update direction
    delta_x = np.zeros(len(x))

    # boolean indicating whether a search should be carried out
    search = True

    # check whether we must make a minimum step anyway, since we're close to
    # the next bound
    next_x_par = x[par_index] + par_direction * options.min_step_size
    if par_direction is -1:
        if next_x_par < problem.lb_full[par_index]:
            delta_x[par_index] = problem.lb_full[par_index] - x[par_index]
            search = False
    else:
        if next_x_par > problem.ub_full[par_index]:
            delta_x[par_index] = problem.ub_full[par_index] - x[par_index]
            search = False

    if not search:
        return (current_profile.fval_path[-1], x + delta_x)

    # parameter extrapolation function
    def par_extrapol(step_length):
        return np.max([np.min([x[par_index] + step_length * par_direction,
                               problem.ub_full[par_index]]),
                       problem.lb_full[par_index]])

    # check if this is the first step, compute first guess of next step
    if len(current_profile.fval_path) == 1:
        step_size_guess = options.default_step_size
        delta_obj_value = 0.
    else:
        step_size_guess = np.abs(current_profile.x_path[par_index,-1] -
                                 current_profile.x_path[par_index,-2])
        delta_obj_value = current_profile.fval_path[-1] - global_opt
    next_theta = par_extrapol(step_size_guess)

    # next start point has to be searched
    # compute, where the next objective value which we aim for
    next_obj_target = - np.log(1. - options.delta_ratio_max) \
                      - magic_factor_obj_value * delta_obj_value + \
                      current_profile.fval_path[-1]

    # compute objective at the guessed point
    problem.fix_parameters(par_index, next_theta)
    current_obj = problem.objective(next_theta)

    if next_obj_target < current_obj:
        # The step was rather too long
        stop_search = False
        while not stop_search:
            # Reduce step size of guess
            step_size_guess = \
                np.min([np.max([step_size_guess / options.step_size_factor,
                                options.min_step_size]),
                        options.max_step_size])
            last_theta = copy.copy(next_theta)
            next_theta = par_extrapol(step_size_guess)

            # Check if we crossed a root or reduced to the minimum
            if step_size_guess == options.min_step_size:
                stop_search = True
            else:
                # compute new objective value
                problem.fix_parameters(par_index, next_theta)
                last_obj = copy.copy(current_obj)
                current_obj = problem.objective(next_theta)

                # check for root crossing
                if next_obj_target >= current_obj:
                    stop_search = True

                    # interpolate between the last two steps
                    delta_obj = np.abs(current_obj - last_obj)
                    delta_theta = np.abs(last_theta - next_theta)
                    add_theta = np.abs(last_obj - next_obj_target) * \
                                delta_theta / delta_obj

                    # fix final guess
                    next_theta = last_theta + add_theta

    else:
        # The step was rather too short
        stop_search = False
        while not stop_search:
            step_size_guess = \
                np.min([np.max([step_size_guess * options.step_size_factor,
                                options.min_step_size]),
                        options.max_step_size])
            last_theta = copy.copy(next_theta)
            next_theta = par_extrapol(step_size_guess)

            # Check if we crossed a root or increased to the maximum
            if step_size_guess == options.max_step_size:
                stop_search = True
            else:
                # compute new objective value
                problem.fix_parameters(par_index, next_theta)
                last_obj = copy.copy(current_obj)
                current_obj = problem.objective(next_theta)

                # check for root crossing
                if next_obj_target <= current_obj:
                    stop_search = True

                    # interpolate between the last two steps
                    delta_obj = np.abs(current_obj - last_obj)
                    delta_theta = np.abs(last_theta - next_theta)
                    add_theta = np.abs(last_obj - next_obj_target) * \
                                delta_theta / delta_obj

                    # fix final guess
                    next_theta = last_theta + add_theta

    delta_x = np.zeros(len(x))
    delta_x[par_index] = next_theta - x[par_index]
    # return (last_obj, last_theta)
    return (next_obj_target, x + delta_x)