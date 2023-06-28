import numpy as np

from pypesto import Problem, optimize
from pypesto.engine import MultiProcessEngine, SingleCoreEngine
from pypesto.objective.julia import JuliaObjective, display_source_ipython
from pypesto.objective.julia.petab_jl_importer import PetabJlImporter

# The pyjulia wrapper appears to ignore global noqas, thus per line here


def test_pyjulia_pipeline():
    """Test that a pipeline with julia objective works."""
    # just make sure display function works
    rng = np.random.default_rng(42)

    assert display_source_ipython(  # noqa: S101
        "doc/example/model_julia/LR.jl"
    )

    # define objective
    obj = JuliaObjective(
        module="LR",
        source_file="doc/example/model_julia/LR.jl",
        fun="fun",
        grad="grad",
    )

    n_p = obj.get("n_p")

    # call consistency
    x = rng.normal(size=n_p)
    assert obj.get_fval(x) == obj.get_fval(x)  # noqa: S101
    assert (obj.get_grad(x) == obj.get_grad(x)).all()  # noqa: S101

    # define problem
    lb, ub = [-5.0] * n_p, [5.0] * n_p
    # create 10 random starting points within the bounds
    x_guesses = rng.uniform(lb, ub, size=(10, n_p))
    problem = Problem(obj, lb=lb, ub=ub, x_guesses=x_guesses)

    # optimize
    result = optimize.minimize(problem, engine=SingleCoreEngine(), n_starts=10)

    # use parallelization
    result2 = optimize.minimize(
        problem, engine=MultiProcessEngine(), n_starts=10
    )

    # check results match
    assert np.allclose(  # noqa: S101
        result.optimize_result[0].x, result2.optimize_result[0].x
    )
    # optimal point won't be true parameter
    x_true = obj.get("p_true")
    assert not np.allclose(x_true, result.optimize_result[0].x)  # noqa: S101

    # check with analytical value
    p_opt = obj.get("p_opt")
    assert np.allclose(result.optimize_result[0].x, p_opt)  # noqa: S101


def test_petabJL_interface():
    """Test the interface to PEtab.jl with provided solutions from julia."""
    model_name = "boehm_JProteomeRes2014"
    yaml_file = f"doc/example/{model_name}/{model_name}.yaml"

    importer = PetabJlImporter.from_yaml(yaml_file)

    problem = importer.create_problem(precompile=False)

    parameters = np.genfromtxt(
        'doc/example/boehm_JProteomeRes2014/Boehm_validation/Parameter.csv',
        delimiter=',',
        skip_header=1,
    )

    # check objective function
    obj_ref = np.genfromtxt(
        'doc/example/boehm_JProteomeRes2014/Boehm_validation/ObjectiveValue'
        '.csv',
        delimiter=',',
        skip_header=1,
    )
    obj = problem.objective(parameters, sensi_orders=(0,))

    assert np.allclose(obj, obj_ref)  # noqa: S101

    # check gradient value
    grad_ref = np.genfromtxt(
        'doc/example/boehm_JProteomeRes2014/Boehm_validation/Gradient.csv',
        delimiter=',',
        skip_header=1,
    )
    grad = problem.objective(parameters, sensi_orders=(1,))

    assert np.allclose(grad, grad_ref)  # noqa: S101

    # check hessian value
    hess_ref = np.genfromtxt(
        'doc/example/boehm_JProteomeRes2014/Boehm_validation/Hessian.csv',
        delimiter=',',
        skip_header=1,
    )
    hess = problem.objective(parameters, sensi_orders=(2,))

    assert np.allclose(hess, hess_ref)  # noqa: S101


def test_petabJL_from_module():
    """Test that PEtab.jl is integrated properly."""
    # create objective
    module = "MyPEtabJlModule"
    source_file = "doc/example/conversion_reaction/PEtabJl_module.jl"

    importer = PetabJlImporter(module=module, source_file=source_file)

    problem = importer.create_problem()

    # optimize with single core
    optimize.minimize(problem, engine=SingleCoreEngine(), n_starts=2)
    # optimize with multi core
    optimize.minimize(
        problem, engine=MultiProcessEngine(n_procs=1), n_starts=2
    )


def test_petabJL_from_yaml():
    """Test that PEtab.jl from yaml file is running smoothly."""
    yaml_file = "doc/example/conversion_reaction/conversion_reaction.yaml"

    importer = PetabJlImporter.from_yaml(yaml_file)

    problem = importer.create_problem(precompile=False)

    # optimize with single core
    optimize.minimize(problem, engine=SingleCoreEngine(), n_starts=10)
