import numpy as np


def fixed_step(x, par_index, par_direction, profile_options, current_profile):
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

        current_profile: pypesto.ProfilerResults
            The profile which should be computed
       """
    delta_x = np.zeros(len(x))
    delta_x[par_index] = par_direction * profile_options.step_size

    return x + delta_x


def adaptive_step_order_0(x, par_index, par_direction, profile_options,
                          current_profile):
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

        current_profile: pypesto.ProfilerResults
            The profile which should be computed
       """
    delta_x = np.zeros(len(x))
    delta_x[par_index] = par_direction * step_size

    return x + delta_x