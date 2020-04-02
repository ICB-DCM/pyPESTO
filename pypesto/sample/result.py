import numpy as np
from typing import Iterable, Union


class McmcPtResult(dict):
    """The result of a sampler run using Markov-chain Monte Carlo, and
    optionally parallel tempering.

    Can be used like a dict.

    Parameters
    ----------
    trace_x: [n_chain, n_par, n_iter]
        Parameters
    trace_fval: [n_chain, n_iter]
        Function values.
    temperatures: [n_chain]
        The associated temperatures.
    time:
        Execution time.
    n_fval: int
        Number of function evaluations.
    n_grad: int
        Number of gradient evaluations.
    n_hess: int
        Number of Hessian evaluations.
    message: str
        Textual comment on the profile result.
    """

    def __init__(self,
                 trace_x: np.ndarray,
                 trace_fval: np.ndarray,
                 temperatures: Iterable[float],
                 time: float = 0.0,
                 n_fval: int = 0,
                 n_grad: int = 0,
                 n_hess: int = 0,
                 message: str = None):
        super().__init__()

        self.trace_x = trace_x
        self.trace_fval = trace_fval
        self.temperatures = temperatures
        self.time = time
        self.n_fval = n_fval
        self.n_grad = n_grad
        self.n_hess = n_hess
        self.message = message

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
