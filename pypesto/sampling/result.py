import numpy as np
from typing import Iterable


class McmcPtResult(dict):
    """The result of a sampler run using Markov-chain Monte Carlo, and
    optionally parallel tempering.

    Can be used like a dict.

    Parameters
    ----------
    trace_x: [n_chain, n_iter, n_par]
        Parameters.
    trace_neglogpost: [n_chain, n_iter]
        Function values.
    betas: [n_chain]
        The associated inverse temperatures.
    burn_in: [n_chain]
        The burn in index.
    message: str
        Textual comment on the profile result.

    Here, `n_chain` denotes the number of chains, `n_iter` the number of
    iterations (i.e., the chain length), and `n_par` the number of parameters.
    """

    def __init__(self,
                 trace_x: np.ndarray,
                 trace_neglogpost: np.ndarray,
                 betas: Iterable[float],
                 burn_in: int = None,
                 time: float = 0.,
                 message: str = None):
        super().__init__()

        self.trace_x = trace_x
        self.trace_neglogpost = trace_neglogpost
        self.betas = betas
        self.burn_in = burn_in
        self.time = time
        self.message = message

        if trace_x.ndim != 3:
            raise ValueError(f"trace_x.ndim not as expected: {trace_x.ndim}")
        if trace_neglogpost.ndim != 2:
            raise ValueError("trace_neglogpost.ndim not as expected: "
                             f"{trace_neglogpost.ndim}")
        if trace_x.shape[0] != trace_neglogpost.shape[0] \
                or trace_x.shape[1] != trace_neglogpost.shape[1]:
            raise ValueError("Trace dimensions do not match:"
                             f"trace_x.shape={trace_x.shape},"
                             f"trace_neglogpost.shape={trace_neglogpost.shape}")

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
