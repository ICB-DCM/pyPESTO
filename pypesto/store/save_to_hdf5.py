import os
from typing import Union
from numbers import Integral

import h5py
import numpy as np

from .hdf5 import write_array, write_float_array
from ..result import Result


def check_overwrite(f: Union[h5py.File, h5py.Group],
                    overwrite: bool,
                    target: str):
    """
    Checks whether target already exists. Sends a warning if
    overwrite=False, deletes the target if overwrite=True.

    Attributes
    -------------
    f: file or group where existence of a group with the path group_path
       should be checked
    target: name of the group, whose existence is checked
    overwrite: if overwrite is true, it deltes the target in f
    """
    if target in f:
        if overwrite:
            del f[target]
        else:
            raise Exception(f"The file already exists and contains "
                            f"information about {target} result."
                            f"If you wish to overwrite the file set"
                            f"overwrite=True.")


class ProblemHDF5Writer:
    """
    Writer of the HDF5 problem files.

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
            HDF5 problem file name
        """
        self.storage_filename = storage_filename

    def write(self, problem, overwrite: bool = False):
        """
        Write HDF5 problem file from pyPESTO problem object.
        """

        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, 'problem')
            attrs_to_save = [a for a in dir(problem) if not a.startswith('__')
                             and not callable(getattr(problem, a))
                             and not hasattr(type(problem), a)]

            problem_grp = f.create_group("problem")
            # problem_grp.attrs['config'] = objective.get_config()

            for problem_attr in attrs_to_save:
                value = getattr(problem, problem_attr)
                if isinstance(value, (list, np.ndarray)):
                    value = np.asarray(value)
                    if value.size:
                        write_array(problem_grp, problem_attr, value)
                elif isinstance(value, Integral):
                    problem_grp.attrs[problem_attr] = value


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

        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, 'optimization')
            optimization_grp = get_or_create_group(f, "optimization")
            # settings =
            # optimization_grp.create_dataset("settings", settings, dtype=)
            results_grp = get_or_create_group(optimization_grp, "results")

            for start in result.optimize_result.list:
                start_id = start['id']
                start_grp = get_or_create_group(results_grp, start_id)
                # start['history'] = None  # TOOD temporary fix
                for key in start.keys():
                    if key == 'history':
                        continue
                    if isinstance(start[key], np.ndarray):
                        write_float_array(start_grp, key, start[key])
                    elif start[key] is not None:
                        start_grp.attrs[key] = start[key]
                f.flush()


class SamplingResultHDF5Writer:
    """
    Writer of the HDF5 sampling files.

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

    def write(self, result: Result, overwrite: bool = False):
        """
        Write HDF5 sampling file from pyPESTO result object.
        """

        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, 'sampling')
            sampling_grp = get_or_create_group(f, "sampling")
            results_grp = get_or_create_group(sampling_grp, "results")

            for key in result.sample_result.keys():
                if isinstance(result.sample_result[key], np.ndarray):
                    write_float_array(results_grp,
                                      key,
                                      result.sample_result[key])
                elif result.sample_result[key] is not None:
                    results_grp.attrs[key] = result.sample_result[key]
            f.flush()


class ProfileResultHDF5Writer:
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

    def write(self, result: Result, overwrite: bool = False):
        """
        Write HDF5 result file from pyPESTO result object.
        """

        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, 'profiling')
            profiling_grp = get_or_create_group(f, "profiling")

            for profile_id, profile in enumerate(result.profile_result.list):
                profile_grp = get_or_create_group(profiling_grp,
                                                  str(profile_id))
                for parameter_id, parameter_profile in enumerate(profile):
                    result_grp = get_or_create_group(profile_grp,
                                                     str(parameter_id))

                    if parameter_profile is None:
                        result_grp.attrs['IsNone'] = True
                        continue
                    result_grp.attrs['IsNone'] = False
                    for key in parameter_profile.keys():
                        if isinstance(parameter_profile[key], np.ndarray):
                            write_float_array(result_grp,
                                              key,
                                              parameter_profile[key])
                        elif parameter_profile[key] is not None:
                            result_grp.attrs[key] = parameter_profile[key]
            f.flush()
