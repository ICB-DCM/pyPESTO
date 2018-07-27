import scipy.optimize
import re
import dlib

class Optimizer:

    def __init__(self, solver='SciPy_L-BFGS-B'):

        self.solver = solver
        self.tol = 1e-9
        self.options = {'maxiter': 1000, 'disp': False}

    def minimize(self, problem, x0):

        lb = problem.upper_parameter_bounds
        ub = problem.lower_parameter_bounds

        if re.match('^(?i)(scipy_)',self.solver):

            scipy_method = self.solver[6:]
            bounds = scipy.optimize.Bounds(lb[0, :],ub[0, :])

            res = scipy.optimize.minimize(
                problem.objective.get_fval,
                x0,
                method=scipy_method,
                jac=problem.objective.get_grad,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )

        elif re.match('^(?i)(dlib_)',self.solver):

            dlib_method = self.solver[5:]

            res = dlib.find_min_global(
                problem.objective.get_fval_vararg,
                list(lb[0, :]),
                list(ub[0, :]),
                int(self.options['maxiter']),
                0.002,
            )

        return res