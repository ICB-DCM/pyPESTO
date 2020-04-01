import numpy as np
from typing import Iterable


class McmcPtResult(dict):
    """The result of a sampler run using Markov-chain Monte Carlo, and
    optionally parallel tempering.
    The standardized return value of `pypesto.sample`, which can be
    initialized from an OptimizerResult.

    Can be used like a dict.

    Attributes
    ----------

    chains:
        The chains.
    temperatures:
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

    Notes
    -----

    Any field not supported by the profiler or the profiling optimizer is
    filled with None. Some fields are filled by pypesto itself.
    """

    def __init__(self,
                 x_0: np.ndarray,
                 chains: Iterable[Iterable[np.ndarray]],
                 temperatures: Iterable[float],
                 time: float = 0.0,
                 n_fval: int = 0,
                 n_grad: int = 0,
                 n_hess: int = 0,
                 message: str = None):
        super().__init__()

        self.x_0 = x_0
        self.chains = chains
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
