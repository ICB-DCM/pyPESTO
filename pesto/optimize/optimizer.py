import scipy.optimize
import re

class Optimizer:

    def __init__(self, solver='SciPy_BFGS'):

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
                hess=scipy.optimize.BFGS(),
                bounds=bounds,
                tol=self.tol,
                options=self.options)

        return res