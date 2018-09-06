import logging
import os
import unittest
import pypesto


class LoggingTest(unittest.TestCase):

    def test_optimize(self):
        # logging
        logger = logging.getLogger('pypesto')
        logger.setLevel(logging.DEBUG)
        filename = ".test_logging.tmp"
        if os.path.exists(filename):
            os.remove(filename)
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.info("start test")

        # problem definition
        def fun(x):
            raise Exception("This function cannot be called.")

        objective = pypesto.Objective(fun=fun)
        problem = pypesto.Problem(objective, -1, 1)

        optimizer = pypesto.ScipyOptimizer()
        options = {'allow_failed_starts': True}

        # optimization
        pypesto.minimize(problem, optimizer, 5, options=options)

        # assert logging worked
        self.assertTrue(os.path.exists(filename))
        f = open(filename, 'rb')
        content = str(f.read())
        f.close()
        self.assertTrue("failed" in content)

        # tidy up
        os.remove(filename)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(LoggingTest())
    unittest.main()
