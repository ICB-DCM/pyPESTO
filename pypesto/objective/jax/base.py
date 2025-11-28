"""
Jax models interface.

Adds an interface for the construction of loss functions
incorporating jax models. This permits computation of derivatives using a
combination of objective based methods and jax based autodiff.
"""

import copy
from functools import partial
from typing import Callable, Union

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


def _base_objective_as_jax_array_tuple(func: Callable):
    def decorator(*args, **kwargs):
        # make sure return is a tuple of jax arrays
        results = func(*args, **kwargs)
        if isinstance(results, tuple):
            return tuple(jnp.array(r) for r in results)
        return jnp.array(results)

    return decorator


# jax compatible (jit-able) objective function using external callback, see
# https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html


@partial(custom_jvp, nondiff_argnums=(0,))
def _device_fun(base_objective: ObjectiveBase, x: jnp.array) -> jnp.array:
    """Jax compatible objective function execution using external callback.

    Parameters
    ----------
    obj:
        The wrapped jax objective.
    x:
        jax computed input array.

    Returns
    -------
    fval : jnp.array
        The function value as 0-dimensional jax array.
    """
    return jax.pure_callback(
        _base_objective_as_jax_array_tuple(
            partial(base_objective, sensi_orders=(0,))
        ),
        jax.ShapeDtypeStruct((), x.dtype),
        x,
        vmap_method="sequential",
    )


def _device_fun_value_and_grad(
    base_objective: ObjectiveBase, x: jnp.array
) -> tuple[jnp.array, jnp.array]:
    """Jax compatible objective gradient execution using external callback.

    This function will be called when computing the gradient of the
    `JaxObjective` using `jax.grad` or `jax.value_and_grad`. In the latter
    case, the function will return both the function value and the gradient,
    so no caching is necessary. For higher order derivatives, caching would
    be advantageous, but unclear how to implement this.

    Parameters
    ----------
    obj:
        The wrapped jax objective.
    x:
        jax computed input array.

    Returns
    -------
    fval : jnp.array
        The function value as 0-dimensional jax array.
    grad : jnp.array
        The gradient as jax array.
    """
    return jax.pure_callback(
        _base_objective_as_jax_array_tuple(
            partial(
                base_objective,
                sensi_orders=(
                    0,
                    1,
                ),
            )
        ),
        (
            jax.ShapeDtypeStruct((), x.dtype),
            jax.ShapeDtypeStruct(
                x.shape,  # bootstrap from cached value
                x.dtype,
            ),
        ),
        x,
        vmap_method="sequential",
    )


# define custom jvp for device_fun to enable autodiff, see
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
    """Objective function that enables use of pypesto objectives in jax models.

    The generated function should generally be compatible with jax, but cannot
    compute higher order derivatives and is not vectorized (but still
    compatible with jax.vmap)

    Parameters
    ----------
    objective:
        pyPESTO objective to be wrapped.

    Note
    ----
    Currently only implements MODE_FUN and sensi_orders<=1. Support for
    MODE_RES should be straightforward to add.
    """

    def __init__(
        self,
        objective: ObjectiveBase,
    ):
        if not isinstance(objective, ObjectiveBase):
            raise TypeError("objective must be an ObjectiveBase instance")
        if not objective.check_mode(MODE_FUN):
            raise NotImplementedError(
                f"objective must support mode={MODE_FUN}"
            )
        self.base_objective = objective

        # would be cleaner to also have this as class method, but not supported
        # by signature inspection in bind call.
        self.jax_objective = partial(_device_fun, self.base_objective)

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
                and max(sensi_orders) <= 1
            )

    def __call__(
        self,
        x: jnp.ndarray,
        sensi_orders: tuple[int, ...] = (0,),
        mode: ModeType = MODE_FUN,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[jnp.ndarray, tuple, ResultDict]:
        """
        See :class:`ObjectiveBase` for more documentation.

        Note that this function delegates pre- and post-processing as well as
        history handling to the inner objective.
        """

        if not self.check_mode(mode):
            raise ValueError(
                f"This Objective cannot be called with mode={mode}."
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
        sensi_orders: tuple[int, ...],
        mode: ModeType,
        return_dict: bool,
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
        )
        return other

    @property
    def history(self):
        """Expose the history of the inner objective."""
        return self.base_objective.history

    @property
    def pre_post_processor(self):
        """Expose the pre_post_processor of inner objective."""
        return self.base_objective.pre_post_processor

    @pre_post_processor.setter
    def pre_post_processor(self, new_pre_post_processor):
        """Set the pre_post_processor of inner objective."""
        self.base_objective.pre_post_processor = new_pre_post_processor

    @property
    def x_names(self):
        """Expose the x_names of inner objective."""
        return self.base_objective.x_names
