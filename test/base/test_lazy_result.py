import os
import unittest

import h5py
import numpy as np

from pypesto.result import LazyOptimizerResult, OptimizerResult, Result
from pypesto.store import read_result, write_result

from ..visualize import create_optimization_result


class TestLazyOptimizerResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary HDF5 file with sample data
        cls.filename = "test_optimization_results.h5"
        cls.result = create_optimization_result()

        write_result(cls.result, cls.filename, overwrite=True)

    @classmethod
    def tearDownClass(cls):
        # Remove the temporary file after tests
        if os.path.exists(cls.filename):
            os.remove(cls.filename)

    def setUp(self):
        # Load the results lazily for each test
        self.lazy_results = []
        with h5py.File(self.filename, "r") as f:
            for group_name in f["optimization/results"].keys():
                self.lazy_results.append(
                    LazyOptimizerResult(
                        self.filename, f"optimization/results/{group_name}"
                    )
                )

    def test_initialization(self):
        # Test that the lazy results are initialized correctly
        for lazy_result in self.lazy_results:
            self.assertEqual(lazy_result.filename, self.filename)
            self.assertEqual(lazy_result._data, {})
            self.assertEqual(
                lazy_result.group_name,
                f"optimization/results/{lazy_result.id}",
            )

    def test_lazy_loading(self):
        # Test that data is loaded lazily
        for lazy_result in self.lazy_results:
            self.assertIsNone(lazy_result._data.get("x"))
            x = lazy_result.x
            self.assertIsNotNone(lazy_result._data.get("x"))
            np.testing.assert_array_equal(x, lazy_result._data["x"])

    def test_attribute_access(self):
        # Test accessing attributes
        for lazy_result in self.lazy_results:
            np.testing.assert_array_equal(
                lazy_result.x, lazy_result._data["x"]
            )

    def test_read_result_lazy(self):
        # Test reading results using the lazy loading option
        result = read_result(self.filename, optimize=True, lazy=True)
        self.assertIsInstance(result, Result)
        self.assertTrue(result.optimize_result)

        # Check if the optimize results are instances of LazyOptimizerResult
        for opt_result in result.optimize_result:
            self.assertIsInstance(opt_result, LazyOptimizerResult)

    def test_read_result_non_lazy(self):
        # Test reading results without using the lazy loading option
        result = read_result(self.filename, optimize=True, lazy=False)
        self.assertIsInstance(result, Result)
        self.assertTrue(result.optimize_result)

        # Check if the optimize results are instances of OptimizerResult
        for opt_result in result.optimize_result:
            self.assertIsInstance(opt_result, OptimizerResult)
            self.assertNotIsInstance(opt_result, LazyOptimizerResult)

    def test_access(self):
        # Test access for LazyOptimizerResult
        for lazy_result in self.lazy_results:
            # Test direct access
            x_dict = lazy_result["x"]
            x_attr = lazy_result.x
            np.testing.assert_array_equal(x_dict, x_attr)

            fval_dict = lazy_result["fval"]
            fval_attr = lazy_result.fval
            self.assertEqual(fval_dict, fval_attr)

            id_dict = lazy_result["id"]
            id_attr = lazy_result.id
            self.assertEqual(id_dict, id_attr)

    def test_dict_access_consistency(self):
        # Compare lazy and non-lazy results for consistency
        lazy_result = read_result(self.filename, optimize=True, lazy=True)
        non_lazy_result = read_result(self.filename, optimize=True, lazy=False)

        # Test that both support the same access pattern
        for i in range(len(lazy_result.optimize_result)):
            lazy_x = lazy_result.optimize_result.list[i]["x"]
            non_lazy_x = non_lazy_result.optimize_result.list[i]["x"]
            np.testing.assert_array_equal(lazy_x, non_lazy_x)

            lazy_fval = lazy_result.optimize_result.list[i]["fval"]
            non_lazy_fval = non_lazy_result.optimize_result.list[i]["fval"]
            self.assertEqual(lazy_fval, non_lazy_fval)
