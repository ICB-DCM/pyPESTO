"""
Jax models interface.

Adds an interface for the construction of loss functions
incorporating jax models. This permits computation of derivatives using a
combination of objective based methods and jax based autodiff.
"""

import copy
from functools import partial
from typing import Callable, Sequence, Tuple, Union

import numpy as np

from ...C import MODE_FUN, ModeType
from ..base import ObjectiveBase, ResultDict

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_jvp
except ImportError:
    raise ImportError(
        "Using a jax objective requires an installation of "
        "the python package jax. Please install jax via "
        "`pip install jax jaxlib`."
    ) from None

# jax compatible (jit-able) objective function using external callback, see
# https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
# note that these functions are impure since they rely on cached values


@partial(custom_jvp, nondiff_argnums=(0,))
def _device_fun(obj: "JaxObjective", x: jnp.array):
    """Jax compatible objective function execution using host callback.

    This function does not actually call the underlying objective function,
    but instead extracts cached return values. Thus, it must only be called
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
    return jax.pure_callback(
        partial(obj.base_objective, sensi_orders=(0,)),
        jax.ShapeDtypeStruct((), x.dtype),
        x,
    )


def _device_fun_value_and_grad(obj: "JaxObjective", x: jnp.array):
    """Jax compatible objective gradient execution using host callback.

    This function does not actually call the underlying objective function,
    but instead extracts cached return values. Thus, it must only be called
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
    return jax.pure_callback(
        partial(
            obj.base_objective,
            sensi_orders=(
                0,
                1,
            ),
        ),
        (
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct(
                x.shape,  # bootstrap from cached value
                x.dtype,
            ),
        ),
        x,
    )


# define custom jvp for device_fun & device_fun_grad to enable autodiff, see
# https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html


@_device_fun.defjvp
def _device_fun_jvp(
    obj: "JaxObjective", primals: jnp.array, tangents: jnp.array
):
    """JVP implementation for device_fun."""
    (x,) = primals
    (x_dot,) = tangents
    value, grad = _device_fun_value_and_grad(obj, x)
    return value, grad @ x_dot


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
            raise TypeError("objective must be an ObjectiveBase instance")
        if not objective.check_mode(MODE_FUN):
            raise NotImplementedError(
                f"objective must support mode={MODE_FUN}"
            )
        # store names directly rather than calling __init__ of super class
        # as we can't initialize history as we are exposing the history of the
        # inner objective
        self._x_names = x_names

        self.base_objective = objective

        self.jax_fun = jax_fun

        # would be cleaner to also have this as class method, but not supported
        # by signature inspection in bind call.
        def jax_objective(x):
            # device fun doesn't actually need the value of y, but we need to
            # compute this here for autodiff to work
            y = jax_fun(x)
            return _device_fun(self, y)

        self.jax_objective = jax_objective

    def check_mode(self, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        return mode == MODE_FUN

    def check_sensi_orders(self, sensi_orders, mode: ModeType) -> bool:
        """See `ObjectiveBase` documentation."""
        if not self.check_mode(mode):
            return False
        else:
            return (
                self.base_objective.check_sensi_orders(sensi_orders, mode)
                and max(sensi_orders) == 0
            )

    def __call__(
        self,
        x: jnp.ndarray,
        sensi_orders: Tuple[int, ...] = (0,),
        mode: ModeType = MODE_FUN,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[jnp.ndarray, Tuple, ResultDict]:
        """
        See :class:`ObjectiveBase` for more documentation.

        Note that this function delegates pre- and post-processing as well as
        history handling to the inner objective.
        """

        if not self.check_mode(mode):
            raise ValueError(
                f"This Objective cannot be called with mode" f"={mode}."
            )
        if not self.check_sensi_orders(sensi_orders, mode):
            raise ValueError(
                f"This Objective cannot be called with "
                f"sensi_orders= {sensi_orders} and mode={mode}."
            )

        # this computes all the results from the inner objective, rendering
        # them accessible as cached values for device_fun, etc.
        if kwargs.pop("return_dict", False):
            raise ValueError(
                "return_dict=True is not available for JaxObjective evaluation"
            )

        return self.jax_objective(x)

    def call_unprocessed(
        self,
        x: np.ndarray,
        sensi_orders: Tuple[int, ...],
        mode: ModeType,
        **kwargs,
    ) -> ResultDict:
        """
        See :class:`ObjectiveBase` for more documentation.

        This function is not implemented for JaxObjective as it is not called
        in the override for __call__. However, it's marked as abstract so we
        need to implement it.
        """
        pass

    def __deepcopy__(self, memodict=None):
        other = JaxObjective(
            copy.deepcopy(self.base_objective),
            copy.deepcopy(self.jax_fun),
            copy.deepcopy(self.x_names),
        )
        return other

    @property
    def history(self):
        """Exposes the history of the inner objective."""
        return self.base_objective.history

    @property
    def pre_post_processor(self):
        """Exposes the pre_post_processor of inner objective."""
        return self.base_objective.pre_post_processor
