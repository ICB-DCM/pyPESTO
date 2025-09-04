"""Include functions for saving various results to hdf5."""

from __future__ import annotations

import logging
import os
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from .. import OptimizeResult, OptimizerResult
from ..result import ProfilerResult, Result, SampleResult
from .hdf5 import write_array, write_float_array

logger = logging.getLogger(__name__)


def check_overwrite(f: h5py.File | h5py.Group, overwrite: bool, target: str):
    """
    Check whether target already exists.

    Sends a warning if ``overwrite=False``, deletes the target if
    ``overwrite=True``.

    Attributes
    ----------
    f: file or group where existence of a group with the path group_path
       should be checked
    target: name of the group, whose existence is checked
    overwrite: if ``True``, it deletes the target in ``f``
    """
    if target in f:
        if overwrite:
            del f[target]
        else:
            raise RuntimeError(
                f"File `{f.file.filename}` already exists and contains "
                f"information about {target} result. "
                f"If you wish to overwrite the file, set "
                f"`overwrite=True`."
            )


class ProblemHDF5Writer:
    """
    Writer of the HDF5 problem files.

    Attributes
    ----------
    storage_filename:
        HDF5 result file name
    """

    def __init__(self, storage_filename: str | Path):
        """
        Initialize writer.

        Parameters
        ----------
        storage_filename:
            HDF5 problem file name
        """
        self.storage_filename = str(storage_filename)

    def write(self, problem, overwrite: bool = False):
        """Write HDF5 problem file from pyPESTO problem object."""
        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, "problem")
            attrs_to_save = [
                a
                for a in dir(problem)
                if not a.startswith("__")
                and not callable(getattr(problem, a))
                and not hasattr(type(problem), a)
            ]

            problem_grp = f.create_group("problem")
            # save the configuration
            f["problem/config"] = str(problem.objective.get_config())

            for problem_attr in attrs_to_save:
                value = getattr(problem, problem_attr)
                if isinstance(value, (list, np.ndarray)):
                    value = np.asarray(value)
                    if value.size:
                        write_array(problem_grp, problem_attr, value)
                elif isinstance(value, (Integral, str)):
                    problem_grp.attrs[problem_attr] = value


class OptimizationResultHDF5Writer:
    """
    Writer of the HDF5 result files.

    Attributes
    ----------
    storage_filename:
        HDF5 result file name
    """

    def __init__(self, storage_filename: str | Path):
        """
        Initialize Writer.

        Parameters
        ----------
        storage_filename:
            HDF5 result file name
        """
        self.storage_filename = str(storage_filename)

    def write(
        self,
        result: Result
        | OptimizeResult
        | OptimizerResult
        | list[OptimizerResult],
        overwrite=False,
    ):
        """Write HDF5 result file from pyPESTO result object.

        Parameters
        ----------
        result: Result to be saved.
        overwrite: Boolean, whether already existing results should be
            overwritten. This applies to the whole list of results, not only to
            individual results. See :meth:`write_optimizer_result` for
            incrementally writing a sequence of `OptimizerResult`.
        """
        Path(self.storage_filename).parent.mkdir(parents=True, exist_ok=True)

        if isinstance(result, Result):
            results = result.optimize_result.list
        elif isinstance(result, OptimizeResult):
            results = result.list
        elif isinstance(result, list):
            results = result
        elif isinstance(result, OptimizerResult):
            results = [result]
        else:
            raise ValueError(f"Unsupported type for `result`: {type(result)}.")

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, "optimization")
            optimization_grp = f.require_group("optimization")
            results_grp = optimization_grp.require_group("results")

            for start in results:
                self._do_write_optimizer_result(start, results_grp, overwrite)

    def write_optimizer_result(
        self, result: OptimizerResult, overwrite: bool = False
    ):
        """Write HDF5 result file from pyPESTO result object.

        Parameters
        ----------
        result: Result to be saved.
        overwrite: Boolean, whether already existing results with the same ID
            should be overwritten.s
        """
        Path(self.storage_filename).parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            results_grp = f.require_group("optimization/results")
            self._do_write_optimizer_result(result, results_grp, overwrite)

    def _do_write_optimizer_result(
        self, result: OptimizerResult, g: h5py.Group = None, overwrite=False
    ):
        """Write an OptimizerResult to the given group."""
        sub_group_id = result["id"]
        check_overwrite(g, overwrite, sub_group_id)
        start_grp = g.require_group(sub_group_id)
        for key in result.keys():
            if key == "history":
                continue
            if isinstance(result[key], np.ndarray):
                write_array(start_grp, key, result[key])
            elif result[key] is not None:
                start_grp.attrs[key] = result[key]


class SamplingResultHDF5Writer:
    """
    Writer of the HDF5 sampling files.

    Attributes
    ----------
    storage_filename:
        HDF5 result file name
    """

    def __init__(self, storage_filename: str | Path):
        """
        Initialize Writer.

        Parameters
        ----------
        storage_filename:
            HDF5 result file name
        """
        self.storage_filename = str(storage_filename)

    def write(self, result: Result, overwrite: bool = False):
        """Write HDF5 sampling file from pyPESTO result object."""
        # if there is no sample available, log a warning and return
        # SampleResult is only a dummy class created by the Result class
        # and always indicates the lack of a sampling result.
        if isinstance(result.sample_result, SampleResult):
            logger.warning(
                "Warning: There is no sampling_result, "
                "which you tried to save to hdf5."
            )
            return

        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, "sampling")
            results_grp = f.require_group("sampling/results")

            for key in result.sample_result.keys():
                if isinstance(result.sample_result[key], np.ndarray):
                    write_float_array(
                        results_grp, key, result.sample_result[key]
                    )
                elif result.sample_result[key] is not None:
                    results_grp.attrs[key] = result.sample_result[key]
            f.flush()


class ProfileResultHDF5Writer:
    """
    Writer of the HDF5 result files.

    Attributes
    ----------
    storage_filename:
        HDF5 result file name
    """

    def __init__(self, storage_filename: str | Path):
        """
        Initialize Writer.

        Parameters
        ----------
        storage_filename:
            HDF5 result file name
        """
        self.storage_filename = str(storage_filename)

    def write(self, result: Result, overwrite: bool = False):
        """Write HDF5 result file from pyPESTO result object."""
        # Create destination directory
        if isinstance(self.storage_filename, str):
            basedir = os.path.dirname(self.storage_filename)
            if basedir:
                os.makedirs(basedir, exist_ok=True)

        with h5py.File(self.storage_filename, "a") as f:
            check_overwrite(f, overwrite, "profiling")
            profiling_grp = f.require_group("profiling")

            for profile_id, profile in enumerate(result.profile_result.list):
                profile_grp = profiling_grp.require_group(str(profile_id))
                for parameter_id, parameter_profile in enumerate(profile):
                    result_grp = profile_grp.require_group(str(parameter_id))
                    self._write_profiler_result(parameter_profile, result_grp)

            f.flush()

    @staticmethod
    def _write_profiler_result(
        parameter_profile: ProfilerResult | None, result_grp: h5py.Group
    ) -> None:
        """Write a single ProfilerResult to hdf5.

        Writes a single profile for a single parameter to the provided HDF5 group.
        """
        if parameter_profile is None:
            result_grp.attrs["IsNone"] = True
            return

        result_grp.attrs["IsNone"] = False

        for key, value in parameter_profile.items():
            try:
                if isinstance(value, np.ndarray):
                    write_float_array(result_grp, key, value)
                elif value is not None:
                    result_grp.attrs[key] = value
            except Exception as e:
                raise ValueError(
                    f"Error writing {key} ({value}) to {result_grp}."
                ) from e


def write_result(
    result: Result,
    filename: str | Path,
    overwrite: bool = False,
    problem: bool = True,
    optimize: bool = False,
    profile: bool = False,
    sample: bool = False,
):
    """
    Save whole pypesto.Result to hdf5 file.

    Boolean indicators allow specifying what to save.

    Parameters
    ----------
    result:
        The :class:`pypesto.Result` object to be saved.
    filename:
        The HDF5 filename.
    overwrite:
        Boolean, whether already existing results should be overwritten.
    problem:
        Read the problem.
    optimize:
        Read the optimize result.
    profile:
        Read the profile result.
    sample:
        Read the sample result.
    """
    if not any([optimize, profile, sample]):
        optimize = True
        profile = True
        sample = True

    if problem:
        pypesto_problem_writer = ProblemHDF5Writer(filename)
        pypesto_problem_writer.write(result.problem, overwrite=overwrite)

    if optimize:
        pypesto_opt_writer = OptimizationResultHDF5Writer(filename)
        pypesto_opt_writer.write(result, overwrite=overwrite)

    if profile:
        pypesto_profile_writer = ProfileResultHDF5Writer(filename)
        pypesto_profile_writer.write(result, overwrite=overwrite)

    if sample:
        pypesto_sample_writer = SamplingResultHDF5Writer(filename)
        pypesto_sample_writer.write(result, overwrite=overwrite)

    if hasattr(result, "variational_result"):
        logger.warning(
            "Results from variational inference are not saved in the hdf5 file. "
            "You have to save them manually."
        )
