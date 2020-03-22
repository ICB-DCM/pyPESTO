import numpy as np


class SamplerResult(dict):
    """
    The result of a sampler run. The standardized return return value from
    pypesto.sample, which can be initialized from an OptimizerResult.

    Can be used like a dict.

    Attributes
    ----------

    chains:
        The chains.
    temperatures:
        The associated temperatures.
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
                 x_0,
                 time=0.0,
                 n_fval=0,
                 n_grad=0,
                 n_hess=0,
                 message=None):
        super().__init__()

        # initialize profile path
        x_shape = x_0.shape
        if len(x_shape) == 1:
            self.x_samples = np.zeros((x_shape[0], 0))
            self.x_samples[:, 0] = x_0[:]
        else:
            self.x_samples = np.zeros((x_shape[0], x_shape[1]))
            self.x_samples[:, :] = x_0[:, :]

        self.total_time = time
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

    def append_samples(self,
                       x_samples,
                       time=0.0,
                       n_fval=0,
                       n_grad=0,
                       n_hess=0):
        """
        This function appends a new OptimizerResult to an existing
        ProfilerResults
        """

        # concatenate samples
        self.x_samples = np.concatenate((self.x_samples, x_samples), axis=1)

        # increment the time and f_eval counters
        self.total_time += time
        self.n_fval += n_fval
        self.n_grad += n_grad
        self.n_hess += n_hess
