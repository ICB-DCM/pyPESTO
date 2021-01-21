import h5py
from ..result import Result
from ..optimize.result import OptimizerResult
from ..profile.result import ProfilerResult
from ..sample.result import McmcPtResult
from ..problem import Problem
from ..objective import Objective, ObjectiveBase
import numpy as np
import logging


logger = logging.getLogger(__name__)


def read_hdf5_profile(f: h5py.File,
                      profile_id: str,
                      parameter_id: str) -> 'ProfilerResult':
    """
    Read HDF5 results per start.

    Parameters
    -------------
    f:
        The HDF5 result file
    profile_id:
        specifies the profile start that is read
        from the HDF5 file
    parameter_id:
        specifies the profile index that is read
        from the HDF5 file
    """

    result = ProfilerResult(np.array([]), np.array([]), np.array([]))

    for profile_key in result.keys():
        if profile_key in f[f'/profiling/{profile_id}/{parameter_id}']:
            result[profile_key] = \
                f[f'/profiling/{profile_id}/{parameter_id}/{profile_key}'][:]
        elif profile_key in \
                f[f'/profiling/{profile_id}/{parameter_id}'].attrs:
            result[profile_key] = \
                f[f'/profiling/{profile_id}/{parameter_id}'].attrs[profile_key]
    return result


def read_hdf5_optimization(f: h5py.File,
                           opt_id: str) -> 'OptimizerResult':
    """
    Read HDF5 results per start.

    Parameters
    -------------
    f:
        The HDF5 result file
    opt_id:
        Specifies the start that is read from the HDF5 file
    """

    result = OptimizerResult()

    for optimization_key in result.keys():
        if optimization_key in f[f'/optimization/results/{opt_id}']:
            result[optimization_key] = \
                f[f'/optimization/results/{opt_id}/{optimization_key}'][:]
        elif optimization_key in \
                f[f'/optimization/results/{opt_id}'].attrs:
            result[optimization_key] = \
                f[f'/optimization/results/{opt_id}'].attrs[optimization_key]
    return result


class ProblemHDF5Reader:
    """
    Reader of the HDF5 problem files written
    by class ProblemHDF5Writer.

    Attributes
    -------------
    storage_filename:
        HDF5 problem file name
    """
    def __init__(self, storage_filename: str):
        """
        Parameters
        ----------

        storage_filename: str
            HDF5 problem file name
        """
        self.storage_filename = storage_filename

    def read(self, objective: ObjectiveBase = None) -> Problem:
        """
        Read HDF5 problem file and return pyPESTO problem object.

        Parameters
        ----------
        objective:
            Objective function which is currently not saved to storage.
        Returns
        -------
        problem:
            A problem instance with all attributes read in.
        """
        # create empty problem
        if objective is None:
            objective = Objective()
        problem = Problem(objective, [], [])

        with h5py.File(self.storage_filename, 'r') as f:
            for problem_key in f['/problem']:
                setattr(problem, problem_key,
                        f[f'/problem/{problem_key}'][:])
            for problem_attr in f['/problem'].attrs:
                setattr(problem, problem_attr,
                        f['/problem'].attrs[problem_attr])

        # h5 uses numpy for everything; convert to lists where necessary
        problem.x_fixed_vals = [float(val) for val in problem.x_fixed_vals]
        problem.x_fixed_indices = [int(ix) for ix in problem.x_fixed_indices]
        problem.x_names = [name.decode() for name in problem.x_names]

        return problem


class OptimizationResultHDF5Reader:
    """
    Reader of the HDF5 result files written
    by class OptimizationResultHDF5Writer.

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
        self.results = Result()

    def read(self) -> Result:
        """
        Read HDF5 result file and return pyPESTO result object.
        """
        with h5py.File(self.storage_filename, "r") as f:
            if '/problem' in f['/']:
                problem_reader = ProblemHDF5Reader(self.storage_filename)
                self.results.problem = problem_reader.read()

            for opt_id in f['/optimization/results']:
                result = read_hdf5_optimization(f, opt_id)
                self.results.optimize_result.append(result)
            self.results.optimize_result.sort()
        return self.results


class SamplingResultHDF5Reader:
    """
    Reader of the HDF5 result files written
    by class SamplingResultHDF5Writer.

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
        self.results = Result()

    def read(self) -> Result:
        """
        Read HDF5 result file and return pyPESTO result object.
        """
        sample_result = {}
        with h5py.File(self.storage_filename, "r") as f:
            if '/problem' in f['/']:
                problem_reader = ProblemHDF5Reader(self.storage_filename)
                self.results.problem = problem_reader.read()
            for key in f['/sampling/results']:
                sample_result[key] = \
                    f[f'/sampling/results/{key}'][:]
            for key in f['/sampling/results'].attrs:
                sample_result[key] = \
                    f['/sampling/results'].attrs[key]
        self.results.sample_result = McmcPtResult(**sample_result)

        return self.results


class ProfileResultHDF5Reader:
    """
    Reader of the HDF5 result files written
    by class OptimizationResultHDF5Writer.

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
        self.results = Result()

    def read(self) -> Result:
        """
        Read HDF5 result file and return pyPESTO result object.
        """
        profiling_list = []
        with h5py.File(self.storage_filename, "r") as f:
            if '/problem' in f['/']:
                problem_reader = ProblemHDF5Reader(self.storage_filename)
                self.results.problem = problem_reader.read()
            for profile_id in f['/profiling']:
                profiling_list.append([])
                for parameter_id in f[f'/profiling/{profile_id}']:
                    if f[f'/profiling/{profile_id}/'
                         f'{parameter_id}'].attrs['IsNone']:
                        profiling_list[int(profile_id)].append(None)
                    else:
                        profiling_list[int(profile_id)]\
                            .append(
                            read_hdf5_profile(f,
                                              profile_id=profile_id,
                                              parameter_id=parameter_id))
            self.results.profile_result.list = profiling_list
        return self.results
