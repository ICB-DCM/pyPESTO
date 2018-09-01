import numpy as np
import scipy.optimize
import re
import abc
import time
import os
from .objective import ObjectiveOptimizeState

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
                 x0=None,
                 fval0=None,
                 trace=None,
                 exitflag=None,
                 time=None,
                 message=None):
        super().__init__()
        self.x = np.array(x)
        self.fval = fval
        self.grad = np.array(grad)
        self.hess = np.array(hess)
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.x0 = np.array(x0)
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


<<<<<<< HEAD
def objective_decorator(minimize):
    """
    Default decorator for the minimize() method to initialise and extract
    information stored in the objective

=======
def timed_minimize(minimize):
    """
    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.
>>>>>>> feature_fixedpars
    """
    def timed_minimize(self, problem, x0, index):

        if self.temp_file is not None:
            temp_file = self.temp_file.replace('{index}', str(index))
        else:
            temp_file = None
            
        problem.objective.reset_history()

        problem.objective.reset_history(
            len(x0),
            temp_file,
            self.temp_save_iter,
        )

        result = minimize(self, problem, x0, index)

        result = self.fill_result_from_objective(result, problem.objective)

        if temp_file is not None and os.path.isfile(temp_file):
            os.remove(temp_file)

        return result
    return timed_minimize


def fixed_minimize(minimize):
    """
    Default decorator for the minimize() method to include also fixed
    parameters in the result arrays (nans will be inserted in the
    derivatives).
    """
    def fixed_minimize(self, problem, x0):
        result = minimize(self, problem, x0)
        result.x = problem.get_full_vector(result.x, problem.x_fixed_vals)
        result.grad = problem.get_full_vector(result.grad)
        result.hess = problem.get_full_matrix(result.hess)
        result.x0 = problem.get_full_vector(result.x0, problem.x_fixed_vals)
        return result
    return fixed_minimize


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
        self.temp_file = None
        self.temp_save_iter = 10

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
            x0=startpoint,
            exitflag=-1,
            message='{0}'.format(err),
        )
        self.fill_result_from_objective(result, objective)

        return result

    def fill_result_from_objective(self, result, objective):

        result.x = objective.min_x
        if self.is_least_squares():
            result.fval = objective.get_fval(objective.min_x)
        else:
            result.fval = objective.min_fval

        result.n_fval = objective.n_fval
        result.n_grad = objective.n_grad
        result.n_hess = objective.n_hess

        if objective.trace is not None \
                and len(objective.trace):
            result.fval0 = objective.trace.loc[0].fval

        result.trace = objective.trace
        result.time = objective.start_time - time.time()

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

<<<<<<< HEAD
    @objective_decorator
    def minimize(self, problem, x0, index):
=======
    @fixed_minimize
    @timed_minimize
    def minimize(self, problem, x0):
>>>>>>> feature_fixedpars
        lb = problem.lb
        ub = problem.ub

        if self.is_least_squares():
            # is a residual based least squares method

            if problem.objective.res is None:
                raise Exception('Least Squares optimization is not available '
                                'for this type of objective.')

            ls_method = self.method[3:]
            bounds = (lb, ub)
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
<<<<<<< HEAD
=======
            least_squares = False

>>>>>>> feature_fixedpars
            # is a fval based optimization method
            bounds = scipy.optimize.Bounds(lb, ub)

            fun_may_return_tuple = self.method.lower() in \
                ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
                    'dogleg', 'trust-ncg']
            if fun_may_return_tuple and problem.objective.grad is True:
                def fun(x):
                    return problem.objective.call_mode_fun(x, (0, 1))
            else:
                fun = problem.objective.get_fval

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
        optimizer_result = OptimizerResult(
<<<<<<< HEAD
            grad=res.jac if hasattr(res, 'jac') else None,
            hess=res.hess if hasattr(res, 'hess') else None,
=======
            x=res.x,
            fval=res.fun if not least_squares
            else problem.objective.get_fval(res.x),
            grad=getattr(res, 'jac', None),
            hess=getattr(res, 'hess', None),
            n_fval=getattr(res, 'nfev', 0),
            n_grad=getattr(res, 'njev', 0),
            n_hess=getattr(res, 'nhev', 0),
>>>>>>> feature_fixedpars
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

<<<<<<< HEAD
    @objective_decorator
    def minimize(self, problem, x0, index):
=======
    @fixed_minimize
    @timed_minimize
    def minimize(self, problem, x0):
>>>>>>> feature_fixedpars

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
