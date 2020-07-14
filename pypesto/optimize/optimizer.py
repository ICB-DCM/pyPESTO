import numpy as np
import scipy.optimize
import re
import abc
import time
import logging
from typing import Dict

from ..objective import (OptimizerHistory, HistoryOptions, CsvHistory)
from ..objective.history import HistoryBase
from ..problem import Problem
from .result import OptimizerResult

try:
    import ipopt
except ImportError:
    ipopt = None

try:
    import dlib
except ImportError:
    dlib = None

try:
    import pyswarm
except ImportError:
    pyswarm = None

EXITFLAG_LOADED_FROM_FILE = -99

logger = logging.getLogger(__name__)


def history_decorator(minimize):
    """
    Default decorator for the minimize() method to initialize and extract
    information stored in the history.
    """

    def wrapped_minimize(self, problem, x0, id, allow_failed_starts,
                         history_options=None):
        objective = problem.objective

        # initialize the objective
        objective.initialize()

        # create optimizer history
        if history_options is None:
            history_options = HistoryOptions()
        history = history_options.create_history(
            id=id, x_names=[problem.x_names[ix]
                            for ix in problem.x_free_indices])
        optimizer_history = OptimizerHistory(history=history, x0=x0)

        # plug in history for the objective to record it
        objective.history = optimizer_history

        # perform the actual minimization
        try:
            result = minimize(self, problem, x0, id, history_options)
            objective.history.finalize()
            result.id = id
        except Exception as err:
            if allow_failed_starts:
                logger.error(f'start {id} failed: {err}')
                result = OptimizerResult(
                    x0=x0, exitflag=-1, message=str(err), id=id)
            else:
                raise

        result = fill_result_from_objective_history(result, objective.history)

        # clean up, history is available from result
        objective.history = HistoryBase()

        return result
    return wrapped_minimize


def time_decorator(minimize):
    """
    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.
    """

    def wrapped_minimize(self, problem, x0, id, allow_failed_starts,
                         history_options=None):
        start_time = time.time()
        result = minimize(self, problem, x0, id, allow_failed_starts,
                          history_options)
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

    def wrapped_minimize(self, problem, x0, id, allow_failed_starts,
                         history_options=None):
        # perform the actual optimization
        result = minimize(self, problem, x0, id, allow_failed_starts,
                          history_options)

        # vectors to full vectors
        result.update_to_full(problem)

        logger.info(f"Final fval={result.fval:.4f}, "
                    f"time={result.time:.4f}s, "
                    f"n_fval={result.n_fval}.")

        return result
    return wrapped_minimize


def fill_result_from_objective_history(
        result: OptimizerResult,
        optimizer_history: OptimizerHistory):
    """
    Overwrite function values in the result object with the values recorded in
    the history.
    """
    update_vals = True
    # check history for better values
    # value could be different e.g. if constraints violated
    if optimizer_history.fval_min is not None and result.fval is not None \
            and not np.isclose(optimizer_history.fval_min, result.fval):
        logger.warning(
            "Function values from history and optimizer do not match: "
            f"{optimizer_history.fval_min}, {result.fval}")
        # do not use value from history
        update_vals = False

    if optimizer_history.x_min is not None and result.x is not None \
            and not np.allclose(result.x, optimizer_history.x_min):
        logger.warning(
            "Parameters obtained from history and optimizer do not match: "
            f"{optimizer_history.x_min}, {result.x}")
        # do not use value from history
        update_vals = result.fval is None

    # update optimal values from history if reported function values
    # and parameters coincide
    if update_vals:
        # override values from history if available
        result.x = optimizer_history.x_min
        result.fval = optimizer_history.fval_min
        if optimizer_history.grad_min is not None:
            result.grad = optimizer_history.grad_min
        if optimizer_history.hess_min is not None:
            result.hess = optimizer_history.hess_min
        if optimizer_history.res_min is not None:
            result.res = optimizer_history.res_min
        if optimizer_history.sres_min is not None:
            result.sres = optimizer_history.sres_min

    # initial values
    result.x0 = optimizer_history.x0
    result.fval0 = optimizer_history.fval0

    # counters
    # we only use our own counters here as optimizers may report differently
    result.n_fval = optimizer_history.history.n_fval
    result.n_grad = optimizer_history.history.n_grad
    result.n_hess = optimizer_history.history.n_hess
    result.n_res = optimizer_history.history.n_res
    result.n_sres = optimizer_history.history.n_sres

    # trace
    result.history = optimizer_history.history

    return result


def read_result_from_file(problem: Problem, history_options: HistoryOptions,
                          identifier: str):
    if history_options.storage_file.endswith('.csv'):
        history = CsvHistory(
            file=history_options.storage_file.format(id=identifier),
            options=history_options,
            load_from_file=True
        )
    else:
        raise NotImplementedError()

    opt_hist = OptimizerHistory(
        history, history.get_x_trace(0),
        generate_from_history=True
    )

    result = OptimizerResult(
        id=identifier,
        message='loaded from file',
        exitflag=EXITFLAG_LOADED_FROM_FILE,
        time=max(history.get_time_trace())
    )
    result.id = identifier
    result = fill_result_from_objective_history(result, opt_hist)
    result.update_to_full(problem)

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
            history_options: HistoryOptions = None,
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
        history_options:
            Optimizer history options.
        """

    @abc.abstractmethod
    def is_least_squares(self):
        return False

    def get_default_options(self):
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

        self.options = options
        if self.options is None:
            self.options = ScipyOptimizer.get_default_options(self)
            self.options['ftol'] = tol
        elif self.options is not None and 'ftol' not in self.options:
            self.options['ftol'] = tol

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_options: HistoryOptions = None,
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

            if self.options is not None:
                ls_options = self.options.copy()
                ls_options['verbose'] = 2 if 'disp' in ls_options.keys() \
                                             and ls_options['disp'] else 0
                ls_options.pop('disp', None)
            else:
                ls_options = {}

            # optimize
            res = scipy.optimize.least_squares(
                fun=fun,
                x0=x0,
                method=ls_method,
                jac=jac,
                bounds=bounds,
                tr_solver='exact',
                loss='linear',
                **ls_options
            )
            # extract fval/grad from result, note that fval is not available
            # from least squares solvers
            grad = getattr(res, 'grad', None)
            fval = None
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
                options=self.options,
            )
            # extract fval/grad from result
            grad = getattr(res, 'jac', None)
            fval = res.fun

        # fill in everything known, although some parts will be overwritten
        optimizer_result = OptimizerResult(
            x=np.array(res.x),
            fval=fval,
            grad=grad,
            hess=getattr(res, 'hess', None),
            exitflag=res.status,
            message=res.message
        )

        return optimizer_result

    def is_least_squares(self):
        return re.match(r'(?i)^(ls_)', self.method)

    def get_default_options(self):
        if self.is_least_squares:
            options = {'max_nfev': 1000, 'disp': False}
        else:
            options = {'maxiter': 1000, 'disp': False}
        return options


class IpoptOptimizer(Optimizer):
    """Use IpOpt (https://pypi.org/project/ipopt/) for optimization."""

    def __init__(
            self, options: Dict = None):
        """
        Parameters
        ----------
        options:
            Options are directly passed on to `ipopt.minimize_ipopt`.
        """
        super().__init__()
        self.options = options

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_options: HistoryOptions = None,
    ) -> OptimizerResult:
        objective = problem.objective

        bounds = np.array([problem.lb, problem.ub]).T

        ret = ipopt.minimize_ipopt(
            fun=objective.get_fval,
            x0=x0,
            method=None,  # ipopt does not use this argument for anything
            jac=objective.get_grad,
            hess=None,  # ipopt does not support Hessian yet
            hessp=None,  # ipopt does not support Hessian vector product yet
            bounds=bounds,
            tol=None,  # can be set via options
            options=self.options,
        )

        # the ipopt return object is a scipy.optimize.OptimizeResult
        return OptimizerResult(
            x=ret.x,
            exitflag=ret.status,
            message=ret.message
        )

    def is_least_squares(self):
        return False


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
            self.options = DlibOptimizer.get_default_options(self)

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
            self,
            problem: Problem,
            x0: np.ndarray,
            id: str,
            history_options: HistoryOptions = None,
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

    def get_default_options(self):
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
            history_options: HistoryOptions = None,
    ) -> OptimizerResult:
        lb = problem.lb
        ub = problem.ub
        if pyswarm is None:
            raise ImportError(
                "This optimizer requires an installation of pyswarm.")

        xopt, fopt = pyswarm.pso(
            problem.objective.get_fval, lb, ub, **self.options)

        optimizer_result = OptimizerResult(
            x=np.array(xopt),
            fval=fopt
        )

        return optimizer_result

    def is_least_squares(self):
        return False
