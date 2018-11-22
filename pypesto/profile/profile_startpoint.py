import numpy as np

def fixed_step(x, par_index, par_direction, step_size):
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
       """
    delta_x = np.zeros(len(x))
    delta_x[par_index] = par_direction * step_size

    return x + delta_x

def simple_step(x, par_index, par_direction):
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
       """
    delta_x = np.zeros(len(x))
    step_size = 0.01
    delta_x[par_index] = par_direction * step_size

    return x + delta_x
