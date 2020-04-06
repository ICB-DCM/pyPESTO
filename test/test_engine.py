import numpy as np
import copy
import cloudpickle as pickle

import pypesto
import test.test_objective as test_objective
from test.petab_util import folder_base
import amici


def test_basic():
    for engine in [pypesto.SingleCoreEngine(),
                   pypesto.MultiProcessEngine(n_procs=2),
                   pypesto.MultiThreadEngine(n_threads=8)]:
        _test_basic(engine)


def _test_basic(engine):
    # set up problem
    objective = test_objective.rosen_for_sensi(max_sensi_order=2)['obj']
    lb = 0 * np.ones((1, 2))
    ub = 1 * np.ones((1, 2))
    problem = pypesto.Problem(objective, lb, ub)
    optimizer = pypesto.ScipyOptimizer(options={'maxiter': 10})
    result = pypesto.minimize(
        problem=problem, n_starts=5, engine=engine, optimizer=optimizer)
    assert len(result.optimize_result.as_list()) == 5


def test_petab():
    for engine in [pypesto.SingleCoreEngine(),
                   pypesto.MultiProcessEngine(n_procs=2),
                   pypesto.MultiThreadEngine(n_threads=4)]:
        _test_petab(engine)


def _test_petab(engine):
    petab_importer = pypesto.PetabImporter.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
    objective = petab_importer.create_objective()
    problem = petab_importer.create_problem(objective)
    optimizer = pypesto.ScipyOptimizer(options={'maxiter': 10})
    result = pypesto.minimize(
        problem=problem, n_starts=3, engine=engine, optimizer=optimizer)
    assert len(result.optimize_result.as_list()) == 3


def test_deepcopy_objective():
    """Test copying objectives (needed for MultiProcessEngine)."""
    petab_importer = pypesto.PetabImporter.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
    objective = petab_importer.create_objective()

    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint)

    objective2 = copy.deepcopy(objective)

    # test some properties
    assert objective.amici_model.getParameterIds() \
        == objective2.amici_model.getParameterIds()
    assert objective.amici_solver.getSensitivityOrder() \
        == objective2.amici_solver.getSensitivityOrder()
    assert objective.amici_solver.getSensitivityMethod() \
        == objective2.amici_solver.getSensitivityMethod()
    assert len(objective.edatas) == len(objective2.edatas)

    assert objective.amici_model is not objective2.amici_model
    assert objective.amici_solver is not objective2.amici_solver
    assert objective.steadystate_guesses is not \
        objective2.steadystate_guesses


def test_pickle_objective():
    """Test serializing objectives (needed for MultiThreadEngine)."""
    petab_importer = pypesto.PetabImporter.from_yaml(
        folder_base + "Zheng_PNAS2012/Zheng_PNAS2012.yaml")
    objective = petab_importer.create_objective()

    objective.amici_solver.setSensitivityMethod(
        amici.SensitivityMethod_adjoint)

    objective2 = pickle.loads(pickle.dumps(objective))

    # test some properties
    assert objective.amici_model.getParameterIds() \
        == objective2.amici_model.getParameterIds()
    assert objective.amici_solver.getSensitivityOrder() \
        == objective2.amici_solver.getSensitivityOrder()
    assert objective.amici_solver.getSensitivityMethod() \
        == objective2.amici_solver.getSensitivityMethod()
    assert len(objective.edatas) == len(objective2.edatas)
