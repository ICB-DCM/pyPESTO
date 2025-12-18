"""Test the execution engines."""

import copy
import os

import amici
import benchmark_models_petab as models
import cloudpickle as pickle
import numpy as np

import pypesto
import pypesto.optimize
import pypesto.petab

from ..util import rosen_for_sensi


def test_basic():
    for engine in [
        pypesto.engine.SingleCoreEngine(),
        pypesto.engine.MultiProcessEngine(n_procs=2),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="spawn"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="fork"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="forkserver"),
        pypesto.engine.MultiThreadEngine(n_threads=4),
    ]:
        _test_basic(engine)


def _test_basic(engine):
    # set up problem
    objective = rosen_for_sensi(max_sensi_order=2)["obj"]
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)
    optimizer = pypesto.optimize.ScipyOptimizer(options={"maxiter": 10})
    result = pypesto.optimize.minimize(
        problem=problem,
        n_starts=2,
        engine=engine,
        optimizer=optimizer,
        progress_bar=False,
    )
    assert len(result.optimize_result) == 2


def test_petab():
    for engine in [
        pypesto.engine.SingleCoreEngine(),
        pypesto.engine.MultiProcessEngine(n_procs=2),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="spawn"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="fork"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="forkserver"),
        pypesto.engine.MultiThreadEngine(n_threads=4),
    ]:
        _test_petab(engine)


def _test_petab(engine):
    petab_importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(
            models.MODELS_DIR,
            "Boehm_JProteomeRes2014",
            "Boehm_JProteomeRes2014.yaml",
        )
    )
    problem = petab_importer.create_problem()
    optimizer = pypesto.optimize.ScipyOptimizer(options={"maxiter": 10})
    result = pypesto.optimize.minimize(
        problem=problem,
        n_starts=3,
        engine=engine,
        optimizer=optimizer,
        progress_bar=False,
    )
    assert len(result.optimize_result) == 3


def test_deepcopy_objective():
    """Test copying objectives (needed for MultiProcessEngine)."""
    petab_importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(
            models.MODELS_DIR,
            "Boehm_JProteomeRes2014",
            "Boehm_JProteomeRes2014.yaml",
        )
    )
    factory = petab_importer.create_objective_creator()
    amici_model = factory.create_model()
    amici_model.setSteadyStateSensitivityMode(
        amici.SteadyStateSensitivityMode.integrateIfNewtonFails
    )
    amici_model.setSteadyStateComputationMode(
        amici.SteadyStateComputationMode.integrateIfNewtonFails
    )
    objective = factory.create_objective(model=amici_model)

    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint
    )

    objective2 = copy.deepcopy(objective)

    # test some properties
    assert (
        objective.amici_model.getParameterIds()
        == objective2.amici_model.getParameterIds()
    )
    assert (
        objective.amici_solver.getSensitivityOrder()
        == objective2.amici_solver.getSensitivityOrder()
    )
    assert (
        objective.amici_solver.getSensitivityMethod()
        == objective2.amici_solver.getSensitivityMethod()
    )
    assert len(objective.edatas) == len(objective2.edatas)

    assert objective.amici_model is not objective2.amici_model
    assert objective.amici_solver is not objective2.amici_solver
    assert objective.steadystate_guesses is not objective2.steadystate_guesses


def test_pickle_objective():
    """Test serializing objectives (needed for MultiThreadEngine)."""
    petab_importer = pypesto.petab.PetabImporter.from_yaml(
        os.path.join(
            models.MODELS_DIR,
            "Boehm_JProteomeRes2014",
            "Boehm_JProteomeRes2014.yaml",
        )
    )
    factory = petab_importer.create_objective_creator()
    objective = factory.create_objective()

    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint
    )

    objective2 = pickle.loads(pickle.dumps(objective))

    # test some properties
    assert (
        objective.amici_model.getParameterIds()
        == objective2.amici_model.getParameterIds()
    )
    assert (
        objective.amici_solver.getSensitivityOrder()
        == objective2.amici_solver.getSensitivityOrder()
    )
    assert (
        objective.amici_solver.getSensitivityMethod()
        == objective2.amici_solver.getSensitivityMethod()
    )
    assert len(objective.edatas) == len(objective2.edatas)


def test_multiprocess_result_order():
    """Test that MultiProcessEngine returns results in correct order.

    This is important because the engine uses as_completed() internally,
    which doesn't guarantee order. The engine must track indices to
    return results in the same order as input tasks.
    """
    for engine in [
        pypesto.engine.MultiProcessEngine(n_procs=2),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="spawn"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="fork"),
        pypesto.engine.MultiProcessEngine(n_procs=2, method="forkserver"),
    ]:
        _test_multiprocess_result_order(engine)


def _test_multiprocess_result_order(engine):
    """Helper function to test result ordering for a given engine."""
    # Set up problem
    objective = rosen_for_sensi(max_sensi_order=2)["obj"]
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)
    optimizer = pypesto.optimize.ScipyOptimizer(options={"maxiter": 10})

    # Define specific starting points with different x0 values
    # This helps us verify ordering by checking the starting points
    n_starts = 5
    startpoints = [
        np.array([0.1 * i, 0.1 * i]) for i in range(1, n_starts + 1)
    ]

    # Run optimization with specific starting points
    result = pypesto.optimize.minimize(
        problem=problem,
        n_starts=n_starts,
        startpoint_method=lambda *args, **kwargs: startpoints.pop(0),
        engine=engine,
        optimizer=optimizer,
        progress_bar=False,
    )

    # Verify we got all results
    assert len(result.optimize_result) == n_starts

    # Verify results are in order by checking the IDs
    # IDs should be sequential: '0', '1', '2', '3', '4'
    for i, opt_result in enumerate(result.optimize_result):
        assert opt_result.id == str(i), (
            f"Result at index {i} has ID {opt_result.id}, expected {str(i)}. "
            "Results are not in the correct order."
        )
