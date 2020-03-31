import h5py
from ..result import OptimizeResult
from ..optimize.result import OptimizerResult


def read_hdf5_optimization(f: h5py.File,
                           start: h5py.Group) -> 'OptimizerResult':
    result = OptimizerResult()
    for optimization_key in f[f'/optimization/results/{start}']:
        if optimization_key in result.keys():
            result[optimization_key] = \
                f[f'/optimization/results/{start}/{optimization_key}']
    return result


class OptimizationResultHDF5Reader:

    def __init__(self, storage_filename):
        self.storage_filename = storage_filename
        self.results = OptimizeResult()

    def read(self) -> 'OptimizeResult':
        with h5py.File(self.storage_filename, "r") as f:
            for start in f['/optimization/results']:
                result = read_hdf5_optimization(f, start)
                self.results.append(result)
        return self.results
