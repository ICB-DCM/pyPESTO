import scipy.optimize
import re
import abc
import time


class OptimizerResult(dict):
    """
    The result of an optimizer run. Used as a standardized return value to
    map from the individual result objects returned by the employed
    optimizers to the format understood by pypesto.

    Can be used like a dict.

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
                 x0=None,
                 fval0=None,
                 x_trace=None,
                 fval_trace=None,
                 exitflag=None,
                 time=None,
                 message=None):
        super().__init__()
        self.x = x
        self.fval = fval
        self.grad = grad
        self.hess = hess
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.x0 = x0
        self.fval0 = fval0
        self.x_trace = x_trace
        self.fval_trace = fval_trace
        self.exitflag = exitflag
        self.time = time
        self.message = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def time_decorator(minimize):
    """
    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.

    """
    def timed_minimize(self, problem, x0):
        start_time = time.time()
        result = minimize(self, problem, x0)
        used_time = time.time() - start_time
        result.time = used_time
        return result
    return timed_minimize


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

    @time_decorator
    def minimize(self, problem, x0):
        lb = problem.lb
        ub = problem.ub

        if re.match('^(?i)(ls_)', self.method):
            # is a residual based least squares method

            least_squares = True

            ls_method = self.method[3:]
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
                verbose=2 if 'disp' in
                self.options.keys() and self.options['disp']
                else 0,
            )

        else:

            least_squares = False

            # is a fval based optimization method
            bounds = scipy.optimize.Bounds(lb[0, :], ub[0, :])

            res = scipy.optimize.minimize(
                problem.objective.get_fval,
                x0,
                method=self.method,
                jac=problem.objective.get_grad,
                hess=problem.objective.get_hess,
                hessp=problem.objective.get_hessp,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )

        # some fields are not filled by all optimizers, then fill in None
        optimizer_result = OptimizerResult(
            x=res.x,
            fval=res.fun if not least_squares
            else problem.objective.get_fval(res.x),
            grad=res.jac if hasattr(res, 'jac') else None,
            hess=res.hess if hasattr(res, 'hess') else None,
            n_fval=res.nfev if hasattr(res, 'nfev') else 0,
            n_grad=res.njev if hasattr(res, 'njev') else 0,
            n_hess=res.nhev if hasattr(res, 'nhev') else 0,
            x0=x0,
            fval0=None,
            exitflag=res.status,
            message=res.message
        )

        return optimizer_result

    @staticmethod
    def get_default_options():
        options = {'maxiter': 1000, 'disp': False}
        return options


class DlibOptimizer(Optimizer):
    """
    Use the Dlib toolbox for optimization.

    TODO: I don't know which optimizers we want to support here.
    """

    def __init__(self, method, options=None):
        super().__init__()

        self.method = method

        self.options = options
        if self.options is None:
            self.options = DlibOptimizer.get_default_options()

    @time_decorator
    def minimize(self, problem, x0):

        try:
            import dlib
        except ImportError:
            print('This optimizer requires an installation of dlib.')

        # dlib requires variable length arguments
        def get_fval_vararg(*par):
            return problem.objective.get_fval(par)

        res = dlib.find_min_global(
            get_fval_vararg,
            list(problem.lb[0, :]),
            list(problem.ub[0, :]),
            int(self.options['maxiter']),
            0.002,
        )

        optimizer_result = OptimizerResult(
            x=res[0],
            fval=res[1],
            x0=x0
        )

        return optimizer_result

    @staticmethod
    def get_default_options():
        return {}
