import scipy.optimize
import re
import sys
import abc


class OptimizerResult
    """
    The results of a (local) optimizer run.

    Attributes
    ----------

    x: ndarray
        The best found parameters.

    fval: float
        The best found function value, fun(x).

    grad, hess: ndarray
        The gradient and Hessian at x.

    n_fval: int
        Number of function evaluations.

    n_grad: int
        Number of gradient evaluations.

    n_hess: int
        Number of Hessian evaluations.

    exitflag: int
        The exitflag of the optimizer.

    message: str
        Textual comment on the optimization result.

    Any field not supported by the optimizer is filled with None.

    """

    def __init__(self,
                 x=None,
                 fval=None,
                 grad=None,
                 hess=None,
                 n_fval=None,
                 n_grad=None,
                 n_hess=None,
                 exitflag=None,
                 message=None):
        self.x = x
        self.fval = fval
        self.grad = grad
        self.hess = hess
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.exitflag = exitflag
        self.message= message


class Optimizer(abc.ABC):
    """
    This is the optimizer base class, not functional on its own.

    An optimizer takes a problem, and possibly a start point, and then
    performs an optimization. It returns an OptimizerResult.
    """

    def __init__(self):
        """
        Default constructor.
        """

    @abc.abstractmethod
    def minimize(self, problem, x0):
        """"
        Perform optimization.
        """

    @staticmethod
    def get_default_options():
        """
        Create default options specific for the optimizer.
        """
        return None


class ScipyOptimizer(Optimizer):
    """
    Use the SciPy optimizers.
    """

    def __init__(self, method='L-BFGS-B', tol=1e-9, options=None):
        super().__init__()

        self.method = method

        self.tol = tol

        self.options = options
        if self.options is None:
            self.options = ScipyOptimizer.get_default_options()

    def minimize(self, problem, x0):

        lb = problem.lb
        ub = problem.ub

        if re.match('^(?i)(ls_)', method):
            ls_method = method[3:]
            bounds = (lb[0, :], ub[0, :])

            res = scipy.optimize.least_squares(
                problem.objective.get_res,
                x0,
                method=ls_method,
                jac=problem.objective.get_sres,
                bounds=bounds,
                ftol=self.tol,
                tr_solver='exact',
                loss='linear',
                )

        else:
            bounds = scipy.optimize.Bounds(lb[0, :], ub[0, :])

            res = scipy.optimize.minimize(
                problem.objective.get_fval,
                x0,
                method=method,
                jac=problem.objective.get_grad,
                hess=problem.objective.get_hess,
                hessp=problem.objective.get_hessp,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
                )

        return OptimizerResult(
            x=res.x,
            fval=res.fun,
            grad=res.jac,
            hess=res.hess,
            n_fval=res.nfev,
            n_grad=res.njev,
            n_hess=res.nhev,
            exitflag=res.status,
            message=res.message
        )

    @staticmethod
    def get_default_options():
        options = {'maxiter': 1000, 'disp': False}
        return options


class DlibOptimizer(Optimizer):
    """
    Use the Dlib toolbox for optimization.

    TODO: I don't know which optimizers we want to suport here.
    """

    def __init__(self, method, options=None):
        super().__init__()

        self.method = method
        self.options = options
        if self.options is None:
            self.options.DlibOptimizer.get_default_options()

    def minimize(self, problem, x0):

        if 'dlib' not in sys.modules:
            try:
                import dlib
            except ImportError:
                print('This optimizer requires an installation of dlib.')

        # dlib requires variable length arguments
        def get_fval_vararg(*par):
            return problem.objective.get_fval(par)

        res = dlib.find_min_global(
            get_fval_vararg,
            list(lb[0, :]),
            list(ub[0, :]),
            int(self.options['maxiter']),
            0.002,
            )

        return OptimizerResult(
            x=res[0],
            fval=res[1]
        )

    @staticmethod
    def get_default_options():
        return {}
