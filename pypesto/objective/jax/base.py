"""
Jax models interface.

Adds an interface for the construction of loss functions
incorporating aesara models. This permits computation of derivatives using a
combination of objective based methods and aesara based backpropagation.
"""

import copy
from typing import Optional, Sequence, Tuple, Callable

import numpy as np
from functools import partial

from ...C import FVAL, GRAD, HESS, MODE_FUN, RDATAS, ModeType
from ..base import ObjectiveBase, ResultDict

try:
    import jax
    import jax.numpy as jnp
    from jax import core, grad, custom_jvp
    import jax.experimental.host_callback as hcb
except ImportError:
    raise ImportError(
        "Using a jax objective requires an installation of "
        "the python package jax. Please install aesara via "
        "`pip install jax jaxlib`."
    )

# jax compatible (jittable) objective function using host callback, see
# https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html


@partial(custom_jvp, nondiff_argnums=(0,))
def device_fun(obj, x):
    return hcb.call(
        obj.cached_fval, x,
        result_shape=jax.ShapeDtypeStruct(
            (),
            np.float64
        ),
    )


@partial(custom_jvp, nondiff_argnums=(0,))
def device_fun_grad(obj, x):
    return hcb.call(
        obj.cached_grad, x,
        result_shape=jax.ShapeDtypeStruct(
            obj.cached_base_ret[GRAD].shape,
            np.float64
        ),
    )


def device_fun_hess(obj, x):
    return hcb.call(
        obj.cached_hess, x,
        result_shape=jax.ShapeDtypeStruct(
            obj.cached_base_ret[HESS].shape,
            np.float64
        ),
    )


def device_fun_jvp(obj, primals, tangents):
    x, = primals
    x_dot, = tangents
    return device_fun(obj, x), device_fun_grad(obj, x).dot(x_dot)


def device_fun_grad_jvp(obj, primals, tangents):
    x, = primals
    x_dot, = tangents
    return device_fun_grad(obj, x), device_fun_hess(obj, x).dot(x_dot)


device_fun.defjvp(device_fun_jvp)
device_fun_grad.defjvp(device_fun_grad_jvp)


class JaxObjective(ObjectiveBase):
    """
    Wrapper around an ObjectiveBase.

    Computes the gradient at each evaluation, caching it for later calls.
    Caching is only enabled after the first time the gradient is asked for
    and disabled whenever the cached gradient is not used, in order not to
    increase computation time for derivative-free samplers.

    Parameters
    ----------
    objective:
        The `pypesto.ObjectiveBase` to wrap.
    jax_fun:
        Aesara function that maps `aet_x` to the variables of `objective`
    """

    def __init__(
        self,
        objective: ObjectiveBase,
        jax_fun: Callable,
        x_names: Sequence[str] = None,
    ):
        if not isinstance(objective, ObjectiveBase):
            raise TypeError('objective must be an ObjectiveBase instance')
        if not objective.check_mode(MODE_FUN):
            raise NotImplementedError(
                f'objective must support mode={MODE_FUN}'
            )
        super().__init__(x_names)
        self.base_objective = objective

        self.jax_fun = jax_fun

        def jax_objective(x):
            y = jax_fun(x)
            return device_fun(self, y)

        self.jax_objective = jax.jit(jax_objective)
        self.jax_objective_grad = jax.jit(grad(jax_objective))
        self.jax_objective_hess = jax.jit(jax.hessian(jax_objective))

        # compiled input mapping
        self.infun = jax.jit(self.jax_fun)

        # temporary storage for evaluation results of objective
        self.cached_base_ret: ResultDict = {}

    def cached_fval(self, _):
        """Return cached function value."""
        return self.cached_base_ret[FVAL]

    def cached_grad(self, _):
        """Return cached gradient."""
        return self.cached_base_ret[GRAD]

    def cached_hess(self, _):
        """Return cached Hessian."""
        return self.cached_base_ret[HESS]

    def check_mode(self, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        return mode == MODE_FUN

    def check_sensi_orders(self, sensi_orders, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        if not self.check_mode(mode):
            return False
        else:
            return self.base_objective.check_sensi_orders(sensi_orders, mode)

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        **kwargs,
    ) -> ResultDict:
        """
        See `ObjectiveBase` for more documentation.

        Main method to overwrite from the base class. It handles and
        delegates the actual objective evaluation.
        """
        # derivative computation in jax always requires lower order derivatives
        if 2 in sensi_orders:
            sensi_orders = (0, 1, 2)
        elif 1 in sensi_orders:
            sensi_orders = (0, 1)
        else:
            sensi_orders = (0,)

        # this computes all the results from the inner objective, rendering
        # them accessible as cached values for device_fun & co
        set_return_dict, return_dict = (
            'return_dict' in kwargs,
            kwargs.pop('return_dict', False),
        )
        self.cached_base_ret = self.base_objective(
            self.infun(x), sensi_orders, mode, return_dict=True, **kwargs
        )
        if set_return_dict:
            kwargs['return_dict'] = return_dict
        ret = {}
        if RDATAS in self.cached_base_ret:
            ret[RDATAS] = self.cached_base_ret[RDATAS]
        if 0 in sensi_orders:
            ret[FVAL] = float(self.jax_objective(x))
        if 1 in sensi_orders:
            ret[GRAD] = self.jax_objective_grad(x)
        if 2 in sensi_orders:
            ret[HESS] = self.jax_objective_hess(x)

        return ret

    def __deepcopy__(self, memodict=None):
        other = JaxObjective(
            copy.deepcopy(self.base_objective),
            self.jax_fun,
            self.x_names,
        )

        return other
