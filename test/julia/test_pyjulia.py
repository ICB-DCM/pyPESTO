from julia.api import Julia

# one way of making pyjulia work, see
#  https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
Julia(compiled_modules=False)

import numpy as np

from pypesto import Problem, optimize
from pypesto.engine import MultiProcessEngine, SingleCoreEngine
from pypesto.objective.julia import JuliaObjective, display_source_ipython

# The pyjulia wrapper appears to ignore global noqas, thus per line here


def test_pyjulia_pipeline():
    """Test that a pipeline with julia objective works."""
    # just make sure display function works
    assert display_source_ipython(  # noqa: S101
        "doc/example/model_julia/SIR.jl"
    )

    # define objective
    obj = JuliaObjective(
        module="SIR",
        source_file="doc/example/model_julia/SIR.jl",
        fun="fun",
        grad="grad",
    )

    # call consistency
    x = np.array([-4.0, -2.0])
    assert obj.get_fval(x) == obj.get_fval(x)  # noqa: S101
    assert (obj.get_grad(x) == obj.get_grad(x)).all()  # noqa: S101

    # gradient check
    x_true = obj.get("p_true")
    assert obj.check_gradients_match_finite_differences(x_true)  # noqa: S101

    # define problem
    lb, ub = [-5.0, -3.0], [-3.0, -1.0]
    problem = Problem(obj, lb=lb, ub=ub)

    # optimize
    result = optimize.minimize(problem, engine=SingleCoreEngine())

    # use parallelization
    result2 = optimize.minimize(problem, engine=MultiProcessEngine())

    # check results match
    assert np.allclose(  # noqa: S101
        result.optimize_result[0].x, result2.optimize_result[0].x
    )
    # optimal point won't be true parameter
    assert not np.allclose(x_true, result.optimize_result[0].x)  # noqa: S101
