import h5py
from ..result import Result
from ..optimize.result import OptimizerResult
from ..problem import Problem
from ..objective import Objective


def read_hdf5_optimization(f: h5py.File,
                           opt_id: h5py.Group) -> 'OptimizerResult':
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
        self.problem = Problem(Objective(), [], [])

    def read(self) -> Problem:
        """
        Read HDF5 problem file and return pyPESTO problem object.
        """
        with h5py.File(self.storage_filename, 'r') as f:
            for problem_key in f['/problem']:
                setattr(self.problem, problem_key,
                        f[f'/problem/{problem_key}'][:])
            for problem_attr in f['/problem'].attrs:
                setattr(self.problem, problem_attr,
                        f['/problem'].attrs[problem_attr])
        return self.problem


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
