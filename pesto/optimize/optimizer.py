import scipy.optimize
import re
import sys
import abc


class OptimizerResult(dict):
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

    Any field not supported by the optimizer is to be filled with None.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__

    __delattr__ = dict.__delitem__

class Optimizer(abc.ABC):
    """
    This is the optimizer base class, not functional on its own.

    An optimizer takes a problem, and possibly a start point, and then
    performs and optimization. It returns an OptimizerResult, which is then
    integrated into a Result object.
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

    @classmethod
    def get_default_options():
        """
        Create default options specific to the optimizer.
        """
        return {}


class ScipyOptimizer(Optimizer):

    def __init__(self, method='L-BFGS-B', tol=1e-9, options=None):
        super().__init__()

        self.method = method

        self.tol = 1e-9

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
            n_hess=res.nhv,
            exitflag=res.status,
            message=res.message
        )

    @classmethod
    def get_default_options():
        options = {'maxiter': 1000, 'disp': False}
        return options


class DlibOptimizer(Optimizer):

    def __init__(self, method, options):
        super().__init__()

        self.method = method
        self.options = options

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

        return res

    def get_default_options():
