import h5py
from ..result import OptimizeResult
from ..optimize.result import OptimizerResult


def read_hdf5_optimization(f: h5py.File,
                           start: h5py.Group) -> 'OptimizerResult':
    """
    Read HDF5 results per start.

    Parameters
    ----------
    f:
        The HDF5 result file
    start:
        Specifies the start that is read from the HDF5 file
    """

    result = OptimizerResult()

    for optimization_key in f[f'/optimization/results/{start}']:
        if optimization_key in result.keys():
            result[optimization_key] = \
                f[f'/optimization/results/{start}/{optimization_key}']
    return result


class OptimizationResultHDF5Reader:
    """
    Reader of the HDF5 result files written by class OptimizationResultHDF5Writer.

    Attributes
    ---------
    storage_filename:
        HDF5 result file name
    """
    def __init__(self, storage_filename: str):
        self.storage_filename = storage_filename
        self.results = OptimizeResult()

    def read(self) -> 'OptimizeResult':
        """
        Read HDF5 result file and return pyPESTO result object.
        """
        with h5py.File(self.storage_filename, "r") as f:
            for start in f['/optimization/results']:
                result = read_hdf5_optimization(f, start)
                self.results.append(result)
        return self.results
