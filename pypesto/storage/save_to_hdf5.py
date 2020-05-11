import h5py
import numpy as np
from typing import Union
from .hdf5 import write_string_array, write_int_array, write_float_array
from ..result import Result


class ProblemHDF5Writer:
    """
    Writer of the HDF5 problem files.

    Attributes
    -------------
    storage_filename:
        HDF5 result file name
    """
    LB = 'lb'
    UB = 'ub'
    LB_FULL = 'lb_full'
    UB_FULL = 'ub_full'
    X_FIXED_VALS = 'x_fixed_vals'
    X_FIXED_INDICES = 'x_fixed_indices'
    X_FREE_INDICES = 'x_free_indices'
    X_NAMES = 'x_names'
    DIM = 'dim'
    DIM_FULL = 'dim_full'

    def __init__(self, storage_filename: str):
        """
        Parameters
        ----------

        storage_filename: str
            HDF5 problem file name
        """
        self.storage_filename = storage_filename

    def write(self, problem, overwrite: bool = False):
        """
        Write HDF5 problem file from pyPESTO problem object.
        """
        with h5py.File(self.storage_filename, "a") as f:
            if "problem" in f:
                if overwrite:
                    del f["problem"]
                else:
                    raise Exception("The file already exists and contains "
                                    "information about optimization result."
                                    "If you wish to overwrite the file set"
                                    "overwrite=True.")

            problem_grp = f.create_group("problem")
            # problem_grp.attrs['config'] = objective.get_config()
            problem_grp.attrs[self.DIM] = problem.dim
            problem_grp.attrs[self.DIM_FULL] = problem.dim_full

            write_float_array(problem_grp, self.LB, problem.lb)
            write_float_array(problem_grp, self.UB, problem.ub)
            write_float_array(problem_grp, self.LB_FULL, problem.lb_full)
            write_float_array(problem_grp, self.UB_FULL, problem.ub_full)
            write_float_array(problem_grp, self.X_FIXED_VALS,
                              problem.x_fixed_vals)
            write_int_array(problem_grp, self.X_FIXED_INDICES,
                            problem.x_fixed_indices)
            write_int_array(problem_grp, self.X_FREE_INDICES,
                            problem.x_free_indices)
            write_string_array(problem_grp, self.X_NAMES, problem.x_names)


def get_or_create_group(f: Union[h5py.File, h5py.Group],
                        group_path: str) -> h5py.Group:
    """
    Helper function that returns a group object for the group with group_path
    relative to f. Creates it if it doesn't exist.

    Attributes
    -------------
    f: file or group where existence of a group with the path group_path
       should be checked
    group_path: the path or simply the name of the group that should exist in f

    Returns
    -------
    grp:
        hdf5 group object with specified path.
    """
    if group_path in f:
        grp = f[group_path]
    else:
        grp = f.create_group(group_path)
    return grp


class OptimizationResultHDF5Writer:
    """
    Writer of the HDF5 result files.

    Attributes
    -------------
    storage_filename:
        HDF5 result file name
    """

    def __init__(self, storage_filename: str):
        """
        Parameters
        ----------

        storage_filename: str
            HDF5 result file name
        """
        self.storage_filename = storage_filename

    def write(self, result: Result, overwrite=False):
        """
        Write HDF5 result file from pyPESTO result object.
        """
        with h5py.File(self.storage_filename, "a") as f:
            optimization_grp = get_or_create_group(f, "optimization")
            # settings =
            # optimization_grp.create_dataset("settings", settings, dtype=)
            results_grp = get_or_create_group(optimization_grp, "results")

            for start in result.optimize_result.list:
                start_id = start['id']
                start_grp = get_or_create_group(results_grp, start_id)
                start['history'] = None  # TOOD temporary fix
                if not overwrite:
                    for key in start.keys():
                        if key in start_grp.keys() or key in start_grp.attrs:
                            raise Exception("The file already exists and "
                                            "contains information about "
                                            "optimization result. If you wish "
                                            "to overwrite it, set "
                                            "overwrite=True.")
                for key in start.keys():
                    if isinstance(start[key], np.ndarray):
                        write_float_array(start_grp, key, start[key])
                    elif start[key] is not None:
                        start_grp.attrs[key] = start[key]
                f.flush()
