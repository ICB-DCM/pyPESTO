import scipy.optimize


class Optimizer:

    def __init__(self, solver='scipy_BFGS'):

        self.solver = solver
        self.tol = 1e-9
        self.options = {'maxiter': 1000, 'disp': False}

    def minimize(self, problem, x0):

        lb = problem.upper_parameter_bounds
        ub = problem.lower_parameter_bounds

        res = scipy.optimize.minimize(
            problem.objective.get_fval,
            x0,
            method='BFGS',
            jac=problem.objective.get_grad,
            bounds=zip(lb, ub),
            tol=self.tol,
            options=self.options)

        return res