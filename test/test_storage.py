"""
This is for testing the pypesto.Storage.
"""
import tempfile
import numpy as np
from .visualize.test_visualize import create_problem, \
    create_optimization_result
from pypesto.storage import ProblemHDF5Writer, OptimizationResultHDF5Writer, \
    OptimizationResultHDF5Reader


class TestResultStorage:

    def test_storage(self):
        problem = create_problem()
        minimize_result = create_optimization_result()
        with tempfile.TemporaryDirectory(dir=f".") as tmpdirname:
            fn = tempfile.mktemp(".hdf5", dir=f"{tmpdirname}")
            problem_writer = ProblemHDF5Writer(fn)
            problem_writer.write(problem)
            opt_result_writer = OptimizationResultHDF5Writer(fn)
            opt_result_writer.write(minimize_result)
            opt_result_reader = OptimizationResultHDF5Reader(fn)
            read_result = opt_result_reader.read()
            for i, opt_res in enumerate(minimize_result.optimize_result.list):
                for key in opt_res:
                    if isinstance(opt_res[key], np.ndarray):
                        np.testing.assert_array_equal(
                            opt_res[key],
                            read_result.optimize_result.list[i][key])
                    else:
                        assert opt_res[key] == \
                               read_result.optimize_result.list[i][key]
