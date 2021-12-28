import os
import sys
import unittest
import numpy as np
import warnings
import re

import pypesto
import pypesto.optimize

from ..util import load_amici_objective


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

optimizers = {
    'scipy': [
        'Nelder-Mead', 'Powell', 'CG', 'BFGS', 'Newton-CG',
        'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
        'trust-ncg', 'trust-exact', 'trust-krylov',
        'ls_trf', 'ls_dogbox'],
    # disabled: ,'trust-constr', 'ls_lm', 'dogleg'
    'pyswarm': ['']
}

ATOL = 1e-2
RTOL = 1e-3


class AmiciObjectiveTest(unittest.TestCase):

    def runTest(self):
        for example in ['conversion_reaction']:
            objective, model = load_amici_objective(example)
            x0 = np.array(list(model.getParameters()))

            df = objective.check_grad(
                x0, eps=1e-3, verbosity=0,
                mode=pypesto.C.MODE_FUN)
            print("relative errors MODE_FUN: ", df.rel_err.values)
            print("absolute errors MODE_FUN: ", df.abs_err.values)
            assert np.all((df.rel_err.values < RTOL) |
                          (df.abs_err.values < ATOL))

            df = objective.check_grad(
                x0, eps=1e-3, verbosity=0,
                mode=pypesto.C.MODE_RES)
            print("relative errors MODE_RES: ", df.rel_err.values)
            print("absolute errors MODE_RES: ", df.rel_err.values)
            assert np.all((df.rel_err.values < RTOL) |
                          (df.abs_err.values < ATOL))

            for library in optimizers.keys():
                for method in optimizers[library]:
                    for fp in [[], [1]]:
                        with self.subTest(library=library,
                                          solver=method,
                                          fp=fp):
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                parameter_estimation(
                                    objective, library, method, fp, 2)


def parameter_estimation(
        objective, library, solver, fixed_pars, n_starts):

    if re.match(r'(?i)^(ls_)', solver):
        options = {
            'max_nfev': 10
        }
    else:
        options = {
            'maxiter': 10
        }

    if library == 'scipy':
        optimizer = pypesto.optimize.ScipyOptimizer(method=solver,
                                                    options=options)
    elif library == 'pyswarm':
        optimizer = pypesto.optimize.PyswarmOptimizer(options=options)
    else:
        raise ValueError("This code should not be reached")

    optimizer.temp_file = os.path.join('test', 'tmp_{index}.csv')

    dim = len(objective.x_ids)
    lb = -2 * np.ones((1, dim))
    ub = 2 * np.ones((1, dim))
    pars = objective.amici_model.getParameters()
    problem = pypesto.Problem(objective, lb, ub,
                              x_fixed_indices=fixed_pars,
                              x_fixed_vals=[pars[idx] for idx in fixed_pars])

    optimize_options = pypesto.optimize.OptimizeOptions(
        allow_failed_starts=False,
    )

    startpoints = pypesto.startpoint.UniformStartpoints(check_fval=True)

    pypesto.optimize.minimize(
        problem, optimizer, n_starts,
        startpoint_method=startpoints,
        options=optimize_options,
        filename=None,
    )


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(AmiciObjectiveTest())
    unittest.main()
