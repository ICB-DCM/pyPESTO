"""
This is for testing optimization of the pypesto.Objective.
"""


import numpy as np
import pypesto
import unittest
import test.test_objective as test_objective
import warnings
import re

optimizers = {
    'scipy': ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
              'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
              'trust-ncg', 'trust-exact', 'trust-krylov',
              'ls_trf', 'ls_dogbox'],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    'dlib': ['default']
}


class OptimizerTest(unittest.TestCase):
    def runTest(self):
        for mode in ['separated', 'integrated']:
            if mode == 'separated':
                obj = test_objective.rosen_for_sensi(max_sensi_order=2,
                                                     integrated=False)['obj']
            elif mode == 'integrated':
                obj = test_objective.rosen_for_sensi(max_sensi_order=2,
                                                     integrated=True)['obj']

            for library in optimizers.keys():
                for method in optimizers[library]:
                    with self.subTest(
                            library=library,
                            solver=method,
                            mode=mode
                    ):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if re.match(r'^(?i)(ls_)', method):
                                # obj has no residuals
                                with self.assertRaises(Exception):
                                    self.check_minimize(
                                        obj, library, method)
                                # no error when allow failed starts
                                self.check_minimize(
                                    obj, library, method, True)
                            else:
                                self.check_minimize(
                                    obj, library, method)

    def check_minimize(self,
                       objective, library, solver, allow_failed_starts=False):

        options = {
            'maxiter': 100
        }

        optimizer = None

        if library == 'scipy':
            optimizer = pypesto.ScipyOptimizer(method=solver,
                                               options=options)
        elif library == 'dlib':
            optimizer = pypesto.DlibOptimizer(method=solver,
                                              options=options)

        lb = 0 * np.ones((1, 2))
        ub = 1 * np.ones((1, 2))
        problem = pypesto.Problem(objective, lb, ub)

        optimize_options = pypesto.OptimizeOptions(
            allow_failed_starts=allow_failed_starts)

        result = pypesto.minimize(
            problem=problem,
            optimizer=optimizer,
            n_starts=1,
            startpoint_method=pypesto.startpoint.uniform,
            options=optimize_options
        )

        self.assertTrue(
            isinstance(result.optimize_result.list[0]['fval'], float))
