"""
Jax models interface.

Adds an interface for the construction of loss functions
incorporating jax models. This permits computation of derivatives using a
combination of objective based methods and jax based autodiff.
"""

import copy
from functools import partial
from typing import Callable, Sequence, Tuple

import numpy as np

from ...C import FVAL, GRAD, HESS, MODE_FUN, RDATAS, ModeType
from ..base import ObjectiveBase, ResultDict

try:
    import jax
    import jax.experimental.host_callback as hcb
    import jax.numpy as jnp
    from jax import custom_jvp, grad
except ImportError:
    raise ImportError(
        "Using a jax objective requires an installation of "
        "the python package jax. Please install jax via "
        "`pip install jax jaxlib`."
    )

# jax compatible (jittable) objective function using host callback, see
# https://jax.readthedocs.io/en/latest/jax.experimental.host_callback.html


@partial(custom_jvp, nondiff_argnums=(0,))
def _device_fun(obj: 'JaxObjective', x: jnp.array):
    """Jax compatible objective function execution using host callback.

    This function does not actually call the underlying objective function,
    but instead extracts cached return values. Thus it must only be called
    from within obj.call_unprocessed, and obj.cached_base_ret must be populated.

    Parameters
    ----------
    obj:
        The wrapped jax objective.
    x:
        jax computed input array.

    Note
    ----
    This function should rather be implemented as class method of JaxObjective,
    but this is not possible at the time of writing as this is not supported
    by signature inspection in the underlying bind call.
    """
    return hcb.call(
        obj.cached_fval,
        x,
        result_shape=jax.ShapeDtypeStruct((), np.float64),
    )


@partial(custom_jvp, nondiff_argnums=(0,))
def _device_fun_grad(obj: 'JaxObjective', x: jnp.array):
    """Jax compatible objective gradient execution using host callback.

    This function does not actually call the underlying objective function,
    but instead extracts cached return values. Thus it must only be called
    from within obj.call_unprocessed and obj.cached_base_ret must be populated.

    Parameters
    ----------
    obj:
        The wrapped jax objective.
    x:
        jax computed input array.

    Note
    ----
    This function should rather be implemented as class method of JaxObjective,
    but this is not possible at the time of writing as this is not supported
    by signature inspection in the underlying bind call.
    """
    return hcb.call(
        obj.cached_grad,
        x,
        result_shape=jax.ShapeDtypeStruct(
            obj.cached_base_ret[GRAD].shape,  # bootstrap from cached value
            np.float64,
        ),
    )


def _device_fun_hess(obj: 'JaxObjective', x: jnp.array):
    """Jax compatible objective Hessian execution using host callback.

    This function does not actually call the underlying objective function,
    but instead extracts cached return values. Thus it must only be called
    from within obj.call_unprocessed and obj.cached_base_ret must be populated.

    Parameters
    ----------
    obj:
        The wrapped jax objective.
    x:
        jax computed input array.

    Note
    ----
    This function should rather be implemented as class method of JaxObjective,
    but this is not possible at the time of writing as this is not supported
    by signature inspection in the underlying bind call.
    """
    return hcb.call(
        obj.cached_hess,
        x,
        result_shape=jax.ShapeDtypeStruct(
            obj.cached_base_ret[HESS].shape,  # bootstrap from cached value
            np.float64,
        ),
    )


# define custom jvp for device_fun & device_fun_grad to enable autodiff, see
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html


@_device_fun.defjvp
def _device_fun_jvp(
    obj: 'JaxObjective', primals: jnp.array, tangents: jnp.array
):
    """JVP implementation for device_fun."""
    (x,) = primals
    (x_dot,) = tangents
    return _device_fun(obj, x), _device_fun_grad(obj, x).dot(x_dot)


@_device_fun_grad.defjvp
def _device_fun_grad_jvp(
    obj: 'JaxObjective', primals: jnp.array, tangents: jnp.array
):
    """JVP implementation for device_fun_grad."""
    (x,) = primals
    (x_dot,) = tangents
    return _device_fun_grad(obj, x), _device_fun_hess(obj, x).dot(x_dot)


class JaxObjective(ObjectiveBase):
    """Objective function that combines pypesto objectives with jax functions.

    The generated objective function will evaluate objective(jax_fun(x)).

    Parameters
    ----------
    objective:
        pyPESTO objective
    jax_fun:
        jax function (not jitted) that computes input to the pyPESTO objective
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

        # would be cleaner to also have this as class method, but not supported
        # by signature inspection in bind call.
        def jax_objective(x):
            # device fun doesn't actually need the value of y, but we need to
            # compute this here for autodiff to work
            y = jax_fun(x)
            return _device_fun(self, y)

        # jit objective & derivatives (not integrated)
        self.jax_objective = jax.jit(jax_objective)
        self.jax_objective_grad = jax.jit(grad(jax_objective))
        self.jax_objective_hess = jax.jit(jax.hessian(jax_objective))

        # jit input function
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
        # derivative computation in jax always requires lower order
        # derivatives, see jvp rules for device_fun and device_fun_grad.
        if 2 in sensi_orders:
            sensi_orders = (0, 1, 2)
        elif 1 in sensi_orders:
            sensi_orders = (0, 1)
        else:
            sensi_orders = (0,)

        # this computes all the results from the inner objective, rendering
        # them accessible as cached values for device_fun, etc.
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
            copy.deepcopy(self.jax_fun),
            copy.deepcopy(self.x_names),
        )

        return other
