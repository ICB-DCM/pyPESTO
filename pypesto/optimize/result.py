import numpy as np

from ..objective import History
from ..problem import Problem


class OptimizerResult(dict):
    """
    The result of an optimizer run. Used as a standardized return value to
    map from the individual result objects returned by the employed
    optimizers to the format understood by pypesto.

    Can be used like a dict.

    Attributes
    ----------
    id:
        Id of the optimizer run. Usually the start index.
    x:
        The best found parameters.
    fval:
        The best found function value, `fun(x)`.
    grad:
        The gradient at `x`.
    hess:
        The Hessian at `x`.
    res:
        The residuals at `x`.
    sres:
        The residual sensitivities at `x`.
    n_fval
        Number of function evaluations.
    n_grad:
        Number of gradient evaluations.
    n_hess:
        Number of Hessian evaluations.
    n_res:
        Number of residuals evaluations.
    n_sres:
        Number of residual sensitivity evaluations.
    x0:
        The starting parameters.
    fval0:
        The starting function value, `fun(x0)`.
    history:
        Objective history.
    exitflag:
        The exitflag of the optimizer.
    time:
        Execution time.
    message: str
        Textual comment on the optimization result.

    Notes
    -----

    Any field not supported by the optimizer is filled with None.
    """

    def __init__(self,
                 id: str = None,
                 x: np.ndarray = None,
                 fval: float = None,
                 grad: np.ndarray = None,
                 hess: np.ndarray = None,
                 res: np.ndarray = None,
                 sres: np.ndarray = None,
                 n_fval: int = None,
                 n_grad: int = None,
                 n_hess: int = None,
                 n_res: int = None,
                 n_sres: int = None,
                 x0: np.ndarray = None,
                 fval0: float = None,
                 history: History = None,
                 exitflag: int = None,
                 time: float = None,
                 message: str = None):
        super().__init__()
        self.id = id
        self.x: np.ndarray = np.array(x) if x is not None else None
        self.fval: float = fval
        self.grad: np.ndarray = np.array(grad) if grad is not None else None
        self.hess: np.ndarray = np.array(hess) if hess is not None else None
        self.res: np.ndarray = np.ndarray(res) if res is not None else None
        self.sres: np.ndarray = np.ndarray(sres) if sres is not None else None
        self.n_fval: int = n_fval
        self.n_grad: int = n_grad
        self.n_hess: int = n_hess
        self.n_res: int = n_res
        self.n_sres: int = n_sres
        self.x0: np.ndarray = np.array(x0) if x0 is not None else None
        self.fval0: float = fval0
        self.history: History = history
        self.exitflag: int = exitflag
        self.time: float = time
        self.message: str = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def update_to_full(self, problem: Problem) -> None:
        """
        Updates values to full vectors/matrices

        Parameters
        ----------
        problem:
            problem which contains info about how to convert to full vectors
            or matrices
        """
        self.x = problem.get_full_vector(self.x, problem.x_fixed_vals)
        self.grad = problem.get_full_vector(self.grad)
        self.hess = problem.get_full_matrix(self.hess)
        self.x0 = problem.get_full_vector(self.x0, problem.x_fixed_vals)
