import h5py
from ..result import Result
from ..optimize.result import OptimizerResult
from ..profile.result import ProfilerResult
from ..sample.result import McmcPtResult
from ..problem import Problem
from ..objective import Objective, ObjectiveBase, Hdf5History
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
                           file_name: str,
                           opt_id: str) -> 'OptimizerResult':
    """
    Read HDF5 results per start.

    Parameters
    -------------
    f:
        The HDF5 result file
    file_name:
        The name of the HDF5 file, needed to create HDF5History
    opt_id:
        Specifies the start that is read from the HDF5 file
    """

    result = OptimizerResult()

    for optimization_key in result.keys():
        if optimization_key == 'history':
            if optimization_key in f:
                result['history'] = Hdf5History(id=opt_id,
                                                file=file_name)
                result['history']._recover_options(file_name)
                continue
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
                result = read_hdf5_optimization(f,
                                                self.storage_filename,
                                                opt_id)
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
        try:
            self.results.sample_result = McmcPtResult(**sample_result)
        except TypeError:
            logger.warning("Warning: You tried loading a non-existent "
                           "sampling result.")

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
                profiling_list.append([
                    None for _ in f[f'/profiling/{profile_id}']
                ])
                for parameter_id in f[f'/profiling/{profile_id}']:
                    if f[f'/profiling/{profile_id}/'
                         f'{parameter_id}'].attrs['IsNone']:
                        continue
                    profiling_list[int(profile_id)][int(parameter_id)] = \
                        read_hdf5_profile(f,
                                          profile_id=profile_id,
                                          parameter_id=parameter_id)
            self.results.profile_result.list = profiling_list
        return self.results


def read_result(filename: str,
                problem: bool = True,
                optimize: bool = False,
                profile: bool = False,
                sample: bool = False,
                ) -> Result:
    """
    This is a function that saves the whole pypesto.Result object in an
    HDF5 file. With booleans one can choose more detailed what to save.

    Parameters
    ----------
    filename:
        The HDF5 filename.
    problem:
        Read the problem.
    optimize:
        Read the optimize result.
    profile:
        Read the profile result.
    sample:
        Read the sample result.

    Returns
    -------
    result:
        Result object containing the results stored in HDF5 file.
    """
    if not any([optimize, profile, sample]):
        optimize = True
        profile = True
        sample = True
    result = Result()

    if problem:
        pypesto_problem_reader = ProblemHDF5Reader(filename)
        result.problem = pypesto_problem_reader.read()

    if optimize:
        pypesto_opt_reader = OptimizationResultHDF5Reader(filename)
        try:
            temp_result = pypesto_opt_reader.read()
            result.optimize_result = temp_result.optimize_result
        except KeyError:
            logger.warning('Loading the optimization result failed. It is '
                           'highly likely that no optimization result exists '
                           f'within {filename}.')

    if profile:
        pypesto_profile_reader = ProfileResultHDF5Reader(filename)
        try:
            temp_result = pypesto_profile_reader.read()
            result.profile_result = temp_result.profile_result
        except KeyError:
            logger.warning('Loading the profiling result failed. It is '
                           'highly likely that no profiling result exists '
                           f'within {filename}.')

    if sample:
        pypesto_sample_reader = SamplingResultHDF5Reader(filename)
        try:
            temp_result = pypesto_sample_reader.read()
            result.sample_result = temp_result.sample_result
        except KeyError:
            logger.warning('Loading the sampling result failed. It is '
                           'highly likely that no sampling result exists '
                           f'within {filename}.')

    return result
