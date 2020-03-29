import numpy as np
import scipy.optimize
import re
import abc
import time
import logging
from typing import Dict

from ..objective import OptimizerHistoryFactory
from ..problem import Problem
from .result import OptimizerResult

try:
    import pyswarm
except ImportError:
    pyswarm = None

try:
    import dlib
except ImportError:
    dlib = None


logger = logging.getLogger(__name__)


def history_decorator(minimize):
    """
    Default decorator for the minimize() method to initialize and extract
    information stored in the history.
    """

    def wrapped_minimize(self, problem, x0, id, history_factory):
        objective = problem.objective

        # plug in history
        objective.history = history_factory.create(
            id=id, x0=x0, x_names=objective.x_names)
        # TODO this can be prettified
        if hasattr(objective, 'reset_steadystate_guesses'):
            objective.reset_steadystate_guesses()

        # perform the actual minimization
        result = minimize(self, problem, x0, id, history_factory)

        objective.history.finalize()
        result.id = id
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

    def wrapped_minimize(self, problem, x0, id, history_factory):
        start_time = time.time()
        result = minimize(self, problem, x0, id, history_factory)
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

    def wrapped_minimize(self, problem, x0, id, history_factory):
        # perform the actual optimization
        result = minimize(self, problem, x0, id, history_factory)

        # vectors to full vectors
        result.x = problem.get_full_vector(result.x, problem.x_fixed_vals)
        result.grad = problem.get_full_vector(result.grad)
        result.hess = problem.get_full_matrix(result.hess)
        result.x0 = problem.get_full_vector(result.x0, problem.x_fixed_vals)

        logger.info(f"Final fval={result.fval:.4f}, "
                    f"time={result.time:.4f}s, "
                    f"n_fval={result.n_fval}.")

        return result
    return wrapped_minimize


def fill_result_from_objective_history(result, history):
    """
    Overwrite function values in the result object with the values recorded in
    the history.
    """
    # id
    result.id = history.id

    # best found values
    result.x = history.x_min
    result.fval = history.fval_min
    result.grad = history.grad_min
    result.hess = history.hess_min
    result.res = history.res_min
    result.sres = history.sres_min

    # initial values
    result.x0 = history.x0
    result.fval0 = history.fval0
    # counters
    result.n_fval = history.n_fval
    result.n_grad = history.n_grad
    result.n_hess = history.n_hess
    result.n_res = history.n_res
    result.n_sres = history.n_sres

    # trace
    result.trace = history.trace

    return result


def recover_result(objective, startpoint, err):
    """
    Upon an error, recover from the objective history whatever available,
    and indicate in exitflag and message that an error occurred.
    """
    result = OptimizerResult(
        x0=startpoint,
        exitflag=-1,
        message=str(err),
    )
    fill_result_from_objective_history(result, objective.history)

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
    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_factory: OptimizerHistoryFactory,
    ) -> OptimizerResult:
        """"
        Perform optimization.

        Parameters
        ----------
        problem:
            The problem to find optimal parameters for.
        x0:
            The starting parameters.
        id:
            Multistart id.
        history_factory:
            Factory to create histories.
        """

    @abc.abstractmethod
    def is_least_squares(self):
        return False

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

    def __init__(self,
                 method: str = 'L-BFGS-B',
                 tol: float = 1e-9,
                 options: Dict = None):
        super().__init__()

        self.method = method

        self.tol = tol

        self.options = options
        if self.options is None:
            self.options = ScipyOptimizer.get_default_options()

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_factory: OptimizerHistoryFactory,
    ) -> OptimizerResult:
        lb = problem.lb
        ub = problem.ub
        objective = problem.objective

        if self.is_least_squares():
            # is a residual based least squares method

            if not objective.has_res:
                raise Exception(
                    "For least squares optimization, the objective "
                    "must be able to compute residuals.")

            ls_method = self.method[3:]
            bounds = (lb, ub)

            fun = objective.get_res
            jac = objective.get_sres if objective.has_sres else '2-point'
            # TODO: pass jac computing methods in options

            # optimize
            res = scipy.optimize.least_squares(
                fun=fun,
                x0=x0,
                method=ls_method,
                jac=jac,
                bounds=bounds,
                ftol=self.tol,
                tr_solver='exact',
                loss='linear',
                verbose=2 if 'disp' in
                self.options.keys() and self.options['disp']
                else 0,
            )

        else:
            # is an fval based optimization method

            if not objective.has_fun:
                raise Exception("For this optimizer, the objective must "
                                "be able to compute function values")

            bounds = scipy.optimize.Bounds(lb, ub)

            # fun_may_return_tuple = self.method.lower() in \
            #    ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
            #        'dogleg', 'trust-ncg']
            # TODO: is it more efficient to have tuple as output of fun?
            method_supports_grad = self.method.lower() in \
                ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
                 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact',
                 'trust-constr']
            method_supports_hess = self.method.lower() in \
                ['newton-cg', 'dogleg', 'trust-ncg', 'trust-krylov',
                 'trust-exact', 'trust-constr']
            method_supports_hessp = self.method.lower() in \
                ['newton-cg', 'trust-ncg', 'trust-krylov', 'trust-constr']

            fun = objective.get_fval
            jac = objective.get_grad \
                if objective.has_grad and method_supports_grad \
                else None
            hess = objective.get_hess \
                if objective.has_hess and method_supports_hess \
                else None
            hessp = objective.get_hessp \
                if objective.has_hessp and method_supports_hessp \
                else None
            # minimize will ignore hessp otherwise
            if hessp is not None:
                hess = None

            # optimize
            res = scipy.optimize.minimize(
                fun=fun,
                x0=x0,
                method=self.method,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                tol=self.tol,
                options=self.options,
            )

        # fill in everything known, although some parts will be overwritten
        optimizer_result = OptimizerResult(
            exitflag=res.status,
            message=res.message
        )

        return optimizer_result

    def is_least_squares(self):
        return re.match(r'(?i)^(ls_)', self.method)

    @staticmethod
    def get_default_options():
        options = {'maxiter': 1000, 'disp': False}
        return options


class DlibOptimizer(Optimizer):
    """
    Use the Dlib toolbox for optimization.
    """

    def __init__(self,
                 method: str,
                 options: Dict = None):
        super().__init__()

        self.method = method

        self.options = options
        if self.options is None:
            self.options = DlibOptimizer.get_default_options()

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_factory: OptimizerHistoryFactory,
    ) -> OptimizerResult:

        lb = problem.lb
        ub = problem.ub
        objective = problem.objective

        if dlib is None:
            raise ImportError(
                "This optimizer requires an installation of dlib."
            )

        if not objective.has_fun:
            raise Exception("For this optimizer, the objective must "
                            "be able to return function values.")

        # dlib requires variable length arguments
        def get_fval_vararg(*x):
            return objective.get_fval(x)

        dlib.find_min_global(
            get_fval_vararg,
            list(lb),
            list(ub),
            int(self.options['maxiter']),
            0.002,
        )

        optimizer_result = OptimizerResult()

        return optimizer_result

    def is_least_squares(self):
        return False

    @staticmethod
    def get_default_options():
        return {}


class PyswarmOptimizer(Optimizer):
    """
    Global optimization using pyswarm.
    """

    def __init__(self, options: Dict = None):
        super().__init__()

        if options is None:
            options = {'maxiter': 200}
        self.options = options

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_factory: OptimizerHistoryFactory,
    ) -> OptimizerResult:
        lb = problem.lb
        ub = problem.ub
        if pyswarm is None:
            raise ImportError(
                "This optimizer requires an installation of pyswarm.")

        pyswarm.pso(problem.objective.get_fval, lb, ub, **self.options)

        optimizer_result = OptimizerResult()

        return optimizer_result

    def is_least_squares(self):
        return False
