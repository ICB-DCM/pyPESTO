import numpy as np
import scipy.optimize
import re
import abc
import time
import os
from ..objective import Objective, res_to_fval

try:
    import dlib
except ImportError:
    dlib = None


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

    Any field not supported by the optimizer is filled with None. Some
    fields are filled by pypesto itself.
    """

    def __init__(self,
                 x=None,
                 fval=None,
                 grad=None,
                 hess=None,
                 n_fval=None,
                 n_grad=None,
                 n_hess=None,
                 n_res=None,
                 n_sres=None,
                 x0=None,
                 fval0=None,
                 trace=None,
                 exitflag=None,
                 time=None,
                 message=None):
        super().__init__()
        self.x = np.array(x)
        self.fval = fval
        self.grad = np.array(grad) if grad is not None else None
        self.hess = np.array(hess) if hess is not None else None
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.n_res = n_res
        self.n_sres = n_sres
        self.x0 = np.array(x0) if x0 is not None else None
        self.fval0 = fval0
        self.trace = trace
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


def objective_decorator(minimize):
    """
    Default decorator for the minimize() method to initialize and extract
    information stored in the objective.
    """
    def wrapped_minimize(self, problem, x0, index):
        problem.objective.reset_history(index=index)
        result = minimize(self, problem, x0 , index)
        result = fill_result_from_objective_history(
            result, problem.objective.history)
        return result
    return wrapped_minimize


def time_decorator(minimize):
    """
    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.
    """
    def wrapped_minimize(self, problem, x0, index):
        start_time = time.time()
        result = minimize(self, problem, x0, index)
        used_time = time.time() - start_time
        result.time = used_time
        return result
    return wrapped_minimize


def fix_decorator(minimize):
    """
    Default decorator for the minimize() method to include also fixed
    parameters in the result arrays (nans will be inserted in the
    derivatives).
    """
    def wrapped_minimize(self, problem, x0, index):
        result = minimize(self, problem, x0, index)
        result.x = problem.get_full_vector(result.x, problem.x_fixed_vals)
        result.grad = problem.get_full_vector(result.grad)
        result.hess = problem.get_full_matrix(result.hess)
        result.x0 = problem.get_full_vector(result.x0, problem.x_fixed_vals)
        return result
    return wrapped_minimize
    
    
def fill_result_from_objective_history(result, history):
    """
    Overwrite function values in the result object with the values recorded in
    the history.
    """

    # counters
    result.n_fval = history.n_fval
    result.n_grad = history.n_grad
    result.n_hess = history.n_hess
    result.n_res = history.n_res
    result.n_sres = history.n_sres

    # initial values
    result.x0 = history.x0
    result.fval0 = history.fval0

    # best found values
    result.x = history.x_min
    result.fval = history.fval_min

    # trace
    result.trace = history.trace

    return result


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
    def minimize(self, problem, x0, index):
        """"
        Perform optimization.
        """

    @abc.abstractmethod
    def is_least_squares(self):
        return False

    def recover_result(self, objective, startpoint, err):
        result = OptimizerResult(
            exitflag=-1,
            message='{0}'.format(err),
        )
        fill_result_from_objective_history(result, objective_history)

        return result

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

    @fix_decorator
    @time_decorator
    @objective_decorator
    def minimize(self, problem, x0, index):
        lb = problem.lb
        ub = problem.ub

        if self.is_least_squares():
            # is a residual based least squares method

            if problem.objective.res is None:
                raise Exception('Least Squares optimization is not available '
                                'for this type of objective.')

            ls_method = self.method[3:]
            bounds = (lb, ub)
            
            # optimize
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
            # is a fval based optimization method
            
            bounds = scipy.optimize.Bounds(lb, ub)

            fun_may_return_tuple = self.method.lower() in \
                ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
                    'dogleg', 'trust-ncg']
            if fun_may_return_tuple and problem.objective.grad is True:
                def fun(x):
                    return problem.objective(x, (0, 1), mode=Objective.MODE_FUN)
            else:
                fun = problem.objective.get_fval
            
            #optimize
            res = scipy.optimize.minimize(
                fun,
                x0,
                method=self.method,
                jac=problem.objective.grad if fun_may_return_tuple else
                problem.objective.get_grad,
                hess=problem.objective.get_hess,
                hessp=problem.objective.get_hessp,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )

        # some fields are not filled by all optimizers, then fill in None
        grad=getattr(res, 'grad', None) if self.is_least_squares() \
            else getattr(res, 'jac', None)
        fval = res_to_fval(res.fun) if self.is_least_squares() \
            else res.fun
            
        optimizer_result = OptimizerResult(           
            x=res.x,
            fval=fval,
            grad=grad,
            hess=getattr(res, 'hess', None),
            n_fval=getattr(res, 'nfev', 0),
            n_grad=getattr(res, 'njev', 0),
            n_hess=getattr(res, 'nhev', 0),
            x0=x0,
            fval0=None,
            exitflag=res.status,
            message=res.message
        )

        return optimizer_result

    def is_least_squares(self):
        return re.match('^(?i)(ls_)', self.method)

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

    @fix_decorator
    @time_decorator
    @objective_decorator
    def minimize(self, problem, x0, index):

        if dlib is None:
            raise ImportError(
                'This optimizer requires an installation of dlib.'
            )

        # dlib requires variable length arguments
        def get_fval_vararg(*par):
            return problem.objective.get_fval(par)

        dlib.find_min_global(
            get_fval_vararg,
            list(problem.lb),
            list(problem.ub),
            int(self.options['maxiter']),
            0.002,
        )

        optimizer_result = OptimizerResult(
            x0=x0
        )

        return optimizer_result

    def is_least_squares(self):
        return False

    @staticmethod
    def get_default_options():
        return {}
