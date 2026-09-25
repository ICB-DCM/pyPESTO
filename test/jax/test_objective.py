"""Test jax-based objective function composition."""

import copy
from functools import partial

import numpy as np
import pytest

import pypesto

from ..util import rosen_for_sensi


@pytest.fixture(params=[True, False])
def integrated(request):
    return request.param


@pytest.fixture(params=[2, 1, 0])
def max_sensi_order(request):
    return request.param


@pytest.mark.parametrize("enable_x64", [True, False])
@pytest.mark.parametrize("fix_parameters", [True, False])
@pytest.mark.flaky(reruns=2)
def test_jax(max_sensi_order, integrated, enable_x64, fix_parameters):
    """Test function composition and gradient computation via jax"""
    import jax
    import jax.numpy as jnp

    if max_sensi_order == 2:
        pytest.skip("Not Implemented")

    jax.config.update("jax_enable_x64", enable_x64)

    from pypesto.objective.jax import JaxObjective
    from pypesto.objective.pre_post_process import FixedParametersProcessor

    prob = rosen_for_sensi(max_sensi_order, integrated, [0, 1])

    x_ref = np.asarray(prob["x"])

    def jax_op_in(x: jnp.array) -> jnp.array:
        # pick a simple function here to avoid numerical issues
        return 3.0 * x

    def jax_op_out(x: jnp.array) -> jnp.array:
        # pick a simple function here to avoid numerical issues
        return 0.5 * x

    # compose rosenbrock function with sinh transformation
    obj = JaxObjective(prob["obj"])

    if fix_parameters:
        obj.pre_post_processor = FixedParametersProcessor(
            dim_full=2,
            x_free_indices=[0],
            x_fixed_indices=[1],
            x_fixed_vals=[0.0],
        )

    # evaluate for a couple of random points such that we can assess
    # compatibility with vmap
    xx = x_ref + np.random.randn(10, x_ref.shape[0])
    if fix_parameters:
        xx = xx[:, obj.pre_post_processor.x_free_indices]

    rvals_ref = [
        jax_op_out(
            prob["obj"](jax_op_in(xxi), sensi_orders=(max_sensi_order,))
        )
        for xxi in xx
    ]

    def _fun(y, pypesto_fun, jax_fun_in, jax_fun_out):
        return jax_fun_out(pypesto_fun(jax_fun_in(y)))

    assert obj.check_sensi_orders((max_sensi_order,), pypesto.C.MODE_FUN)
    assert not obj.check_sensi_orders((max_sensi_order,), pypesto.C.MODE_RES)

    for _obj in (obj, copy.deepcopy(obj)):
        fun = partial(
            _fun,
            pypesto_fun=_obj,
            jax_fun_in=jax_op_in,
            jax_fun_out=jax_op_out,
        )

        if max_sensi_order == 1:
            fun = jax.grad(fun)

        # check compatibility with vmap and jit
        vmapped_fun = jax.vmap(fun)
        rvals_jax = vmapped_fun(xx)
        atol = 0
        # also need to account for roundoff errors in input, so we
        # can't use rtol = 1e-8 for 32bit
        rtol = 1e-16 if enable_x64 else 1e-4
        for x, rref, rj in zip(xx, rvals_ref, rvals_jax, strict=True):
            assert isinstance(rj, jnp.ndarray)
            if max_sensi_order == 0:
                np.testing.assert_allclose(
                    rref, float(rj), atol=atol, rtol=rtol
                )
            if max_sensi_order == 1:
                # g(x) = b(c(x)) => g'(x) = b'(c(x))) * c'(x)
                # f(x) = a(g(x)) => f'(x) = a'(g(x)) * g'(x)
                # c: jax_op_in, b: prob["obj"], a: jax_op_out
                # g(x) = b(c(x))
                g = prob["obj"](jax_op_in(x))
                # g'(x) = b'(c(x))) * c'(x)
                g_prime = prob["obj"](
                    jax_op_in(x), sensi_orders=(1,)
                ) @ jax.jacfwd(jax_op_in)(x)
                # f'(x) = a'(g(x)) * g'(x)
                f_prime = jax.jacfwd(jax_op_out)(g) * g_prime
                np.testing.assert_allclose(
                    f_prime, np.asarray(rj), atol=atol, rtol=rtol
                )
