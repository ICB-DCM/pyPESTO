
"""
This is for testing the petab import.
"""

import unittest


class PetabImportTest(unittest.TestCase):

    def test_compile(self):
        pass

    def test_simulate(self):
        pass

    def test_optimize(self):
        pass

    def test_create_measurement_df_from_rdatas(self):
        pass


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(PetabImportTest())
    unittest.main()
