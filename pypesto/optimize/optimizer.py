import abc
import logging
import re
import time
from typing import Dict, Optional

import numpy as np
import scipy.optimize

from ..C import FVAL, GRAD, MODE_FUN, MODE_RES
from ..objective import HistoryBase, HistoryOptions, OptimizerHistory
from ..problem import Problem
from ..result import OptimizerResult
from .load import fill_result_from_history
from .options import OptimizeOptions

try:
    import cyipopt
except ImportError:
    cyipopt = None

try:
    import dlib
except ImportError:
    dlib = None

try:
    import pyswarm
except ImportError:
    pyswarm = None

try:
    import cma
except ImportError:
    cma = None

try:
    import pyswarms
except ImportError:
    pyswarms = None

try:
    import nlopt
except ImportError:
    nlopt = None

try:
    import fides
    from fides import Optimizer as fidesOptimizer
    from fides.hessian_approximation import HessianApproximation
except ImportError:
    fides = None
    HessianApproximation = None
    fidesOptimizer = None

logger = logging.getLogger(__name__)


def history_decorator(minimize):
    """Initialize and extract information stored in the history.

    Default decorator for the minimize() method.
    """

    def wrapped_minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ):
        if history_options is None:
            history_options = HistoryOptions()

        objective = problem.objective

        # initialize the objective
        objective.initialize()

        history = history_options.create_history(
            id=id,
            x_names=[problem.x_names[ix] for ix in problem.x_free_indices],
        )
        optimizer_history = OptimizerHistory(
            history=history,
            x0=x0,
            lb=problem.lb,
            ub=problem.ub,
        )

        # plug in history for the objective to record it
        objective.history = optimizer_history

        # perform the actual minimization
        try:
            result = minimize(
                self,
                problem=problem,
                x0=x0,
                id=id,
                history_options=history_options,
                optimize_options=optimize_options,
            )
            result.id = id
            objective.history.finalize(
                message=result.message, exitflag=result.exitflag
            )
        except Exception as err:
            if optimize_options.allow_failed_starts:
                logger.error(f'start {id} failed: {err}')
                result = OptimizerResult(
                    x0=x0, exitflag=-1, message=str(err), id=id
                )
            else:
                raise

        # maybe override results from history depending on options
        result = fill_result_from_history(
            result=result,
            optimizer_history=objective.history,
            optimize_options=optimize_options,
        )

        # clean up, history is available from result
        objective.history = HistoryBase()

        return result

    return wrapped_minimize


def time_decorator(minimize):
    """Measure time of optimization.

    Default decorator for the minimize() method to take time.
    Currently, the method time.time() is used, which measures
    the wall-clock time.
    """

    def wrapped_minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ):
        start_time = time.time()
        result = minimize(
            self,
            problem=problem,
            x0=x0,
            id=id,
            history_options=history_options,
            optimize_options=optimize_options,
        )
        used_time = time.time() - start_time
        result.time = used_time
        return result

    return wrapped_minimize


def fix_decorator(minimize):
    """Include also fixed parameters in the result arrays of minimize().

    Default decorator for the minimize() method (nans will be inserted in the
    derivatives).
    """

    def wrapped_minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ):
        # perform the actual optimization
        result = minimize(
            self,
            problem=problem,
            x0=x0,
            id=id,
            history_options=history_options,
            optimize_options=optimize_options,
        )

        # vectors to full vectors
        result.update_to_full(problem)

        logger.debug(
            f"Final fval={result.fval:.4f}, time={result.time:.4f}s, "
            f"n_fval={result.n_fval}.",
        )

        return result

    return wrapped_minimize


class Optimizer(abc.ABC):
    """
    Optimizer base class, not functional on its own.

    An optimizer takes a problem, and possibly a start point, and then
    performs an optimization. It returns an OptimizerResult.
    """

    def __init__(self):
        """Initialize base class."""

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
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """
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
        optimize_options:
            Global optimization options.
        """

    @abc.abstractmethod
    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False

    def get_default_options(self):
        """Create default options specific for the optimizer."""
        return None


def check_finite_bounds(lb, ub):
    """Raise if bounds are not finite."""
    if not np.isfinite(lb).all() or not np.isfinite(ub).all():
        raise ValueError(
            'Selected optimizer cannot work with unconstrained '
            'optimization problems.'
        )


class ScipyOptimizer(Optimizer):
    """
    Use the SciPy optimizers.

    Find details on the optimizer and configuration options at:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.\
        optimize.minimize.html#scipy.optimize.minimize
    """

    def __init__(
        self,
        method: str = 'L-BFGS-B',
        tol: float = None,
        options: Dict = None,
    ):
        super().__init__()

        self.method = method

        self.options = options
        if self.options is None:
            self.options = ScipyOptimizer.get_default_options(self)
        self.tol = tol

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} method={self.method}"
        # print everything that is customized
        if self.tol is not None:
            rep += f" tol={self.tol}"
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        lb = problem.lb
        ub = problem.ub
        objective = problem.objective

        if self.is_least_squares():
            # set tolerance to default of scipy optimizer
            tol = self.tol
            if tol is None:
                tol = 1e-8
            # is a residual based least squares method
            if not objective.has_res:
                raise Exception(
                    "For least squares optimization, the objective "
                    "must be able to compute residuals."
                )

            ls_method = self.method[3:]
            bounds = (lb, ub)

            fun = objective.get_res
            jac = objective.get_sres if objective.has_sres else '2-point'
            # TODO: pass jac computing methods in options

            if self.options is not None:
                ls_options = self.options.copy()
                ls_options['verbose'] = (
                    2
                    if 'disp' in ls_options.keys() and ls_options['disp']
                    else 0
                )
                ls_options.pop('disp', None)
                ls_options['max_nfev'] = ls_options.pop('maxiter', None)
            else:
                ls_options = {}

            # optimize
            res = scipy.optimize.least_squares(
                fun=fun,
                x0=x0,
                method=ls_method,
                jac=jac,
                bounds=bounds,
                tr_solver=ls_options.pop(
                    'tr_solver', 'lsmr' if len(x0) > 1 else 'exact'
                ),
                loss='linear',
                ftol=tol,
                **ls_options,
            )
            # extract fval/grad from result, note that fval is not available
            # from least squares solvers
            grad = getattr(res, 'grad', None)
            fval = None
        else:
            # is an fval based optimization method

            if not objective.has_fun:
                raise Exception(
                    "For this optimizer, the objective must "
                    "be able to compute function values"
                )

            bounds = scipy.optimize.Bounds(lb, ub)

            # fun_may_return_tuple = self.method.lower() in \
            #    ['cg', 'bfgs', 'newton-cg', 'l-bfgs-b', 'tnc', 'slsqp',
            #        'dogleg', 'trust-ncg']
            # TODO: is it more efficient to have tuple as output of fun?
            method_supports_grad = self.method.lower() in [
                'cg',
                'bfgs',
                'newton-cg',
                'l-bfgs-b',
                'tnc',
                'slsqp',
                'dogleg',
                'trust-ncg',
                'trust-krylov',
                'trust-exact',
                'trust-constr',
            ]
            method_supports_hess = self.method.lower() in [
                'newton-cg',
                'dogleg',
                'trust-ncg',
                'trust-krylov',
                'trust-exact',
                'trust-constr',
            ]
            method_supports_hessp = self.method.lower() in [
                'newton-cg',
                'trust-ncg',
                'trust-krylov',
                'trust-constr',
            ]

            fun = objective.get_fval
            jac = (
                objective.get_grad
                if objective.has_grad and method_supports_grad
                else None
            )
            hess = (
                objective.get_hess
                if objective.has_hess and method_supports_hess
                else None
            )
            hessp = (
                objective.get_hessp
                if objective.has_hessp and method_supports_hessp
                else None
            )
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
                tol=self.tol,
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
            message=res.message,
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return re.match(r'(?i)^(ls_)', self.method)

    def get_default_options(self):
        """Create default options specific for the optimizer."""
        if self.is_least_squares():
            options = {'max_nfev': 1000, 'disp': False}
        else:
            options = {'maxiter': 1000, 'disp': False}
        return options


class IpoptOptimizer(Optimizer):
    """Use IpOpt (https://pypi.org/project/ipopt/) for optimization."""

    def __init__(self, options: Dict = None):
        """
        Initialize.

        Parameters
        ----------
        options:
            Options are directly passed on to `cyipopt.minimize_ipopt`.
        """
        super().__init__()
        self.options = options

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        if cyipopt is None:
            raise ImportError(
                "This optimizer requires an installation of ipopt. You can "
                "install ipopt via `pip install ipopt`."
            )

        objective = problem.objective

        bounds = np.array([problem.lb, problem.ub]).T

        ret = cyipopt.minimize_ipopt(
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
            x=ret.x, exitflag=ret.status, message=ret.message
        )

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class DlibOptimizer(Optimizer):
    """Use the Dlib toolbox for optimization."""

    def __init__(self, options: Dict = None):
        super().__init__()

        self.options = options
        if self.options is None:
            self.options = DlibOptimizer.get_default_options(self)
        elif 'maxiter' not in self.options:
            raise KeyError('Dlib options are missing the key word ' 'maxiter.')

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        lb = problem.lb
        ub = problem.ub
        check_finite_bounds(lb, ub)
        objective = problem.objective

        if dlib is None:
            raise ImportError(
                "This optimizer requires an installation of dlib. You can "
                "install dlib via `pip install dlib`."
            )

        if not objective.has_fun:
            raise ValueError(
                "For this optimizer, the objective must "
                "be able to return function values."
            )

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
        """Check whether optimizer is a least squares optimizer."""
        return False

    def get_default_options(self):
        """Create default options specific for the optimizer."""
        return {'maxiter': 10000}


class PyswarmOptimizer(Optimizer):
    """Global optimization using pyswarm."""

    def __init__(self, options: Dict = None):
        super().__init__()

        if options is None:
            options = {'maxiter': 200}
        self.options = options

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        lb = problem.lb
        ub = problem.ub
        if pyswarm is None:
            raise ImportError(
                "This optimizer requires an installation of pyswarm. You can "
                "install pyswarm via `pip install pyswarm."
            )

        check_finite_bounds(lb, ub)

        xopt, fopt = pyswarm.pso(
            problem.objective.get_fval, lb, ub, **self.options
        )

        optimizer_result = OptimizerResult(x=np.array(xopt), fval=fopt)

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class CmaesOptimizer(Optimizer):
    """
    Global optimization using cma-es.

    Package homepage: https://pypi.org/project/cma-es/
    """

    def __init__(self, par_sigma0: float = 0.25, options: Dict = None):
        """
        Initialize.

        Parameters
        ----------
        par_sigma0:
            scalar, initial standard deviation in each coordinate.
            par_sigma0 should be about 1/4th of the search domain width
            (where the optimum is to be expected)
        options:
            Optimizer options that are directly passed on to cma.
        """
        super().__init__()

        if options is None:
            options = {'maxiter': 10000}
        self.options = options
        self.par_sigma0 = par_sigma0

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} par_sigma0={self.par_sigma0}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        lb = problem.lb
        ub = problem.ub

        check_finite_bounds(lb, ub)

        sigma0 = self.par_sigma0 * np.median(ub - lb)
        self.options['bounds'] = [lb, ub]

        if cma is None:
            raise ImportError(
                "This optimizer requires an installation of cma. You can "
                "install cma via `pip install cma."
            )

        result = (
            cma.CMAEvolutionStrategy(
                x0,
                sigma0,
                inopts=self.options,
            )
            .optimize(problem.objective.get_fval)
            .result
        )

        optimizer_result = OptimizerResult(
            x=np.array(result[0]), fval=result[1]
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class ScipyDifferentialEvolutionOptimizer(Optimizer):
    """
    Global optimization using scipy's differential evolution optimizer.

    Package homepage: https://docs.scipy.org/doc/scipy/reference/generated\
        /scipy.optimize.differential_evolution.html

    Parameters
    ----------
    options:
        Optimizer options that are directly passed on to scipy's optimizer.


    Examples
    --------
    Arguments that can be passed to options:

    maxiter:
        used to calculate the maximal number of funcion evaluations by
        maxfevals = (maxiter + 1) * popsize * len(x)
        Default: 100
    popsize:
        population size, default value 15
    """

    def __init__(self, options: Dict = None):
        super().__init__()

        if options is None:
            options = {'maxiter': 100}
        self.options = options

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        bounds = list(zip(problem.lb, problem.ub))

        result = scipy.optimize.differential_evolution(
            problem.objective.get_fval, bounds, **self.options
        )

        optimizer_result = OptimizerResult(
            x=np.array(result.x), fval=result.fun
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class PyswarmsOptimizer(Optimizer):
    """
    Global optimization using pyswarms.

    Package homepage: https://pyswarms.readthedocs.io/en/latest/index.html

    Parameters
    ----------
    par_popsize:
        number of particles in the swarm, default value 10

    options:
        Optimizer options that are directly passed on to pyswarms.
        c1: cognitive parameter
        c2: social parameter
        w: inertia parameter
        Default values are (c1,c2,w) = (0.5, 0.3, 0.9)

    Examples
    --------
    Arguments that can be passed to options:

    maxiter:
        used to calculate the maximal number of funcion evaluations.
        Default: 1000
    """

    def __init__(self, par_popsize: float = 10, options: Dict = None):
        super().__init__()

        all_options = {'maxiter': 1000, 'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        if options is None:
            options = {}
        all_options.update(options)
        self.options = all_options
        self.par_popsize = par_popsize

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} par_popsize={self.par_popsize}"
        # print everything that is customized
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        lb = problem.lb
        ub = problem.ub

        if pyswarms is None:
            raise ImportError(
                "This optimizer requires an installation of pyswarms."
            )

        # check for finite values for the bounds
        if np.isfinite(lb).all() is np.False_:
            raise ValueError(
                "This optimizer can only handle finite lower bounds."
            )
        if np.isfinite(ub).all() is np.False_:
            raise ValueError(
                "This optimizer can only handle finite upper bounds."
            )

        optimizer = pyswarms.single.global_best.GlobalBestPSO(
            n_particles=self.par_popsize,
            dimensions=len(x0),
            options=self.options,
            bounds=(lb, ub),
        )

        def successively_working_fval(swarm: np.ndarray) -> np.ndarray:
            """Evaluate the function for all parameters in the swarm object.

            Parameters
            ----------
            swarm: np.ndarray, shape (n_particales_in_swarm, n_parameters)

            Returns
            -------
            result: np.ndarray, shape (n_particles_in_swarm)
            """
            n_particles = swarm.shape[0]
            result = np.zeros(n_particles)
            # iterate over the particles in the swarm
            for i_particle, par in enumerate(swarm):
                result[i_particle] = problem.objective.get_fval(par)

            return result

        cost, pos = optimizer.optimize(
            successively_working_fval,
            iters=self.options['maxiter'],
            verbose=False,
        )

        optimizer_result = OptimizerResult(
            x=pos,
            fval=float(cost),
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class NLoptOptimizer(Optimizer):
    """
    Global/Local optimization using NLopt.

    Package homepage: https://nlopt.readthedocs.io/en/latest/
    """

    def __init__(
        self,
        method=None,
        local_method=None,
        options: Dict = None,
        local_options: Dict = None,
    ):
        """
        Initialize.

        Parameters
        ----------
        method:
            Local or global Optimizer to use for minimization.
        local_method:
            Local method to use in combination with the global optimizer (
            for the MLSL family of solvers) or to solve a subproblem (for the
            AUGLAG family of solvers)
        options:
            Optimizer options. scipy option `maxiter` is automatically
            transformed into `maxeval` and takes precedence.
        local_options:
            Optimizer options for the local method
        """
        super().__init__()

        if options is None:
            options = {}
        elif 'maxiter' in options:
            options['maxeval'] = options.pop('maxiter')
        if local_options is None:
            local_options = {}
        self.options = options
        self.local_options = local_options

        if nlopt is None:
            raise ImportError(
                "This optimizer requires an installation of NLopt. You can "
                "install NLopt via `pip install nlopt`."
            )

        if method is None:
            method = nlopt.LD_LBFGS

        needs_local_method = [
            nlopt.G_MLSL,
            nlopt.G_MLSL_LDS,
            nlopt.GD_MLSL,
            nlopt.GD_MLSL_LDS,
            nlopt.AUGLAG,
            nlopt.AUGLAG_EQ,
        ]

        if local_method is None and method in needs_local_method:
            local_method = nlopt.LD_LBFGS

        if local_method is not None and method not in needs_local_method:
            raise ValueError(
                f'Method "{method}" does not allow a local '
                f'method. Please set `local_method` to None.'
            )

        self.local_methods = [
            nlopt.LD_VAR1,
            nlopt.LD_VAR2,
            nlopt.LD_TNEWTON_PRECOND_RESTART,
            nlopt.LD_TNEWTON_PRECOND,
            nlopt.LD_TNEWTON_RESTART,
            nlopt.LD_TNEWTON,
            nlopt.LD_LBFGS,
            nlopt.LD_SLSQP,
            nlopt.LD_CCSAQ,
            nlopt.LD_MMA,
            nlopt.LN_SBPLX,
            nlopt.LN_NELDERMEAD,
            nlopt.LN_PRAXIS,
            nlopt.LN_NEWUOA,
            nlopt.LN_NEWUOA_BOUND,
            nlopt.LN_BOBYQA,
            nlopt.LN_COBYLA,
        ]
        self.global_methods = [
            nlopt.GN_ESCH,
            nlopt.GN_ISRES,
            nlopt.GN_AGS,
            nlopt.GD_STOGO,
            nlopt.GD_STOGO_RAND,
            nlopt.G_MLSL,
            nlopt.G_MLSL_LDS,
            nlopt.GD_MLSL,
            nlopt.GD_MLSL_LDS,
            nlopt.GN_CRS2_LM,
            nlopt.GN_ORIG_DIRECT,
            nlopt.GN_ORIG_DIRECT_L,
            nlopt.GN_DIRECT,
            nlopt.GN_DIRECT_L,
            nlopt.GN_DIRECT_L_NOSCAL,
            nlopt.GN_DIRECT_L_RAND,
            nlopt.GN_DIRECT_L_RAND_NOSCAL,
        ]
        self.hybrid_methods = [nlopt.AUGLAG, nlopt.AUGLAG_EQ]
        methods = (
            self.local_methods + self.global_methods + self.hybrid_methods
        )

        if method not in methods:
            raise ValueError(
                f'"{method}" is not a valid method. Valid '
                f'methods are: {methods}'
            )

        self.method = method

        if local_method is not None and local_method not in self.local_methods:
            raise ValueError(
                f'"{local_method}" is not a valid method. Valid '
                f'methods are: {self.local_methods}'
            )

        self.local_method = local_method

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} method={self.method}"
        # print everything that is customized
        if self.local_method is not None:
            rep += f" local_method={self.local_method}"
        if self.options is not None:
            rep += f" options={self.options}"
        if self.local_options is not None:
            rep += f" local_options={self.local_methods}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        opt = nlopt.opt(self.method, problem.dim)

        valid_options = [
            'ftol_abs',
            'ftol_rel',
            'xtol_abs',
            'xtol_rel',
            'stopval',
            'x_weights',
            'maxeval',
            'maxtime',
            'initial_step',
        ]

        def set_options(o, options):
            for option, value in options.items():
                if option not in valid_options:
                    raise ValueError(
                        f'"{option}" is not a valid option. Valid '
                        f'options are: {valid_options}'
                    )
                getattr(o, f'set_{option}')(value)

        if self.local_method is not None:
            local_opt = nlopt.opt(self.local_method, problem.dim)
            set_options(local_opt, self.local_options)
            opt.set_local_optimizer(local_opt)

        if self.method in self.global_methods:
            check_finite_bounds(problem.ub, problem.lb)

        opt.set_lower_bounds(problem.lb)
        opt.set_upper_bounds(problem.ub)

        def nlopt_objective(x, grad):
            if grad.size > 0:
                sensi_orders = (0, 1)
            else:
                sensi_orders = (0,)
            r = problem.objective(x, sensi_orders, MODE_FUN, True)
            if grad.size > 0:
                grad[:] = r[GRAD]  # note that this must be inplace
            return r[FVAL]

        opt.set_min_objective(nlopt_objective)

        set_options(opt, self.options)
        try:
            result = opt.optimize(x0)
            msg = 'Finished Successfully.'
        except (
            nlopt.RoundoffLimited,
            nlopt.ForcedStop,
            ValueError,
            RuntimeError,
            MemoryError,
        ) as e:
            result = None
            msg = str(e)

        optimizer_result = OptimizerResult(
            x=result,
            fval=opt.last_optimum_value(),
            message=msg,
            exitflag=opt.last_optimize_result(),
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False


class FidesOptimizer(Optimizer):
    """
    Global/Local optimization using the trust region optimizer fides.

    Package Homepage: https://fides-optimizer.readthedocs.io/en/latest
    """

    def __init__(
        self,
        hessian_update: Optional['HessianApproximation'] = 'default',
        options: Optional[Dict] = None,
        verbose: Optional[int] = logging.INFO,
    ):
        """
        Initialize.

        Parameters
        ----------
        options:
            Optimizer options.
        hessian_update:
            Hessian update strategy. If this is None, a hybrid approximation
            that switches from the problem.objective provided Hessian (
            approximation) to a BFGS approximation will be used.
        """
        super().__init__()

        if (
            (hessian_update is not None)
            and (hessian_update != 'default')
            and not isinstance(hessian_update, HessianApproximation)
        ):
            raise ValueError(
                'Incompatible type for hessian update. '
                'Must be a HessianApproximation, '
                f'was {type(hessian_update)}.'
            )

        if options is None:
            options = {}

        self.verbose = verbose
        self.options = options
        self.hessian_update = hessian_update

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} "
        # print everything that is customized
        if self.hessian_update is not None:
            if self.hessian_update == 'default':
                rep += f" hessian_update={self.hessian_update}"
            else:
                rep += (
                    f" hessian_update="
                    f"{self.hessian_update.__class__.__name__}"
                )
        if self.verbose is not None:
            rep += f" verbose={self.verbose}"
        if self.options is not None:
            rep += f" options={self.options}"
        return rep + ">"

    @fix_decorator
    @time_decorator
    @history_decorator
    def minimize(
        self,
        problem: Problem,
        x0: np.ndarray,
        id: str,
        history_options: HistoryOptions = None,
        optimize_options: OptimizeOptions = None,
    ) -> OptimizerResult:
        """Perform optimization. Parameters: see `Optimizer` documentation."""
        if fides is None:
            raise ImportError(
                "This optimizer requires an installation of fides. You can "
                "install fides via `pip install fides`."
            )

        if self.hessian_update == 'default':
            if not problem.objective.has_hess:
                logging.warning(
                    'Fides is using BFGS as hessian approximation, '
                    'as the problem does not provide a Hessian. '
                    'Specify a Hessian to use a more efficient '
                    'hybrid approximation scheme.'
                )
                _hessian_update = fides.BFGS()
            else:
                _hessian_update = fides.HybridFixed()
        else:
            _hessian_update = self.hessian_update

        resfun = (
            _hessian_update.requires_resfun
            if _hessian_update is not None
            else False
        )

        args = {'mode': MODE_RES if resfun else MODE_FUN}

        if not problem.objective.has_grad:
            raise ValueError(
                'Fides cannot be applied to problems '
                'with objectives that do not support '
                'gradient evaluation.'
            )

        if _hessian_update is None or (
            _hessian_update.requires_hess and not resfun
        ):
            if not problem.objective.has_hess:
                raise ValueError(
                    'Specified hessian update scheme cannot be '
                    'used with objectives that do not support '
                    'Hessian computation.'
                )
            args['sensi_orders'] = (0, 1, 2)
        else:
            args['sensi_orders'] = (0, 1)

        opt = fides.Optimizer(
            fun=problem.objective,
            funargs=args,
            ub=problem.ub,
            lb=problem.lb,
            verbose=self.verbose,
            hessian_update=_hessian_update,
            options=self.options,
            resfun=resfun,
        )

        try:
            opt.minimize(x0)
            msg = self._convert_exitflag_to_message(opt)
        except RuntimeError as err:
            msg = str(err)

        optimizer_result = OptimizerResult(
            x=opt.x_min,
            fval=opt.fval_min if not resfun else None,
            grad=opt.grad_min,
            hess=opt.hess,
            message=msg,
            exitflag=opt.exitflag,
        )

        return optimizer_result

    def is_least_squares(self):
        """Check whether optimizer is a least squares optimizer."""
        return False

    def _convert_exitflag_to_message(self, opt: fidesOptimizer):
        """
        Convert the exitflag of a run to an informative message.

        Parameters
        ----------
        opt:
            The fides.Optimizer that has finished minimizing storing the
            exitflag.

        Returns
        -------
            An informative message on the cause of termination. Based on
            fides documentation.
        """
        messages = {
            fides.ExitFlag.DID_NOT_RUN: "Optimizer did not run",
            fides.ExitFlag.MAXITER: "Reached maximum number of allowed iterations",
            fides.ExitFlag.MAXTIME: "Expected to reach maximum allowed time in next iteration",
            fides.ExitFlag.NOT_FINITE: "Encountered non-finite fval/grad/hess",
            fides.ExitFlag.EXCEEDED_BOUNDARY: "Exceeded specified boundaries",
            fides.ExitFlag.DELTA_TOO_SMALL: "Trust Region Radius too small to proceed",
            fides.ExitFlag.FTOL: "Converged according to fval difference",
            fides.ExitFlag.XTOL: "Converged according to x difference",
            fides.ExitFlag.GTOL: "Converged according to gradient norm",
        }
        return messages.get(
            opt.exitflag, f"exitflag={opt.exitflag} is not defined in fides."
        )
