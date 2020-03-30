import h5py
import numpy as np
from .hdf5 import write_string_array, write_int_array, write_float_array
from ..result import Result


class ProblemHDF5Writer:
    LB = 'lb'
    UB = 'ub'
    LB_FULL = 'lb_full'
    UB_FULL = 'ub_full'
    X_FIXED_VALUES = 'x_fixed_values'
    X_FIXED_INDICES = 'x_fixed_indices'
    X_NAMES = 'x_names'
    DIM = 'dim'
    DIM_FULL = 'dim_full'

    def __init__(self, storage_filename):
        self.storage_filename = storage_filename

    def write(self, problem):
        with h5py.File(self.storage_filename, "a") as f:
            problem_grp = f.create_group("problem")
            # problem_grp.attrs['config'] = objective.get_config()
            problem_grp.attrs[self.DIM] = problem.dim
            problem_grp.attrs[self.DIM_FULL] = problem.dim_full

            write_float_array(problem_grp, self.LB, problem.lb)
            write_float_array(problem_grp, self.UB, problem.ub)
            write_float_array(problem_grp, self.LB_FULL, problem.lb_full)
            write_float_array(problem_grp, self.UB_FULL, problem.ub_full)
            write_float_array(problem_grp, self.X_FIXED_VALUES,
                              problem.x_fixed_vals)
            write_int_array(problem_grp, self.X_FIXED_INDICES,
                            problem.x_fixed_indices)
            write_string_array(problem_grp, self.X_NAMES, problem.x_names)


class OptimizationResultHDF5Writer:

    def __init__(self, storage_filename):
        self.storage_filename = storage_filename

    def write(self, result: Result):
        with h5py.File(self.storage_filename, "a") as f:
            optimization_grp = f.create_group("optimization")
            # settings =
            # optimization_grp.create_dataset("settings", settings, dtype=)
            results_grp = optimization_grp.create_group("results")
            for i, start in enumerate(result.optimize_result.list):
                start_grp = results_grp.create_group(str(i))

                for key in start.keys():
                    if isinstance(start[key], np.ndarray):
                        write_float_array(start_grp, key, start[key])
                    elif start[key] is not None:
                        start_grp.attrs[key] = start[key]
