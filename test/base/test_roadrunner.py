"""Test the roadrunner interface."""
import copy
import logging
import os

import benchmark_models_petab as models
import numpy as np
import petab
import petabtests
import pytest

import pypesto
import pypesto.objective.roadrunner as objective_rr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "case, model_type, version",
    [
        (case, "sbml", "v1.0.0")
        for case in petabtests.get_cases(format_="sbml", version="v1.0.0")
    ],
)
def test_petab_case(case, model_type, version):
    """Wrapper for _execute_case for handling test outcomes"""
    try:
        _execute_case_rr(case, model_type, version)
    except Exception as e:
        if isinstance(
            e, NotImplementedError
        ) or "Timepoint-specific parameter overrides" in str(e):
            logger.info(
                f"Case {case} expectedly failed. Required functionality is "
                f"not implemented: {e}"
            )
            pytest.skip(str(e))
        else:
            raise e


def _execute_case_rr(case, model_type, version):
    """Run a single PEtab test suite case"""
    case = petabtests.test_id_str(case)
    logger.info(f"Case {case}")

    # case folder
    case_dir = petabtests.get_case_dir(case, model_type, version)

    # load solution
    solution = petabtests.load_solution(
        case, format=model_type, version=version
    )
    gt_llh = solution[petabtests.LLH]
    gt_simulation_dfs = solution[petabtests.SIMULATION_DFS]
    tol_llh = solution[petabtests.TOL_LLH]
    tol_simulations = solution[petabtests.TOL_SIMULATIONS]

    # import petab problem
    yaml_file = case_dir / petabtests.problem_yaml_name(case)

    importer = objective_rr.PetabImporterRR.from_yaml(yaml_file)
    petab_problem = importer.petab_problem
    obj = importer.create_objective()

    # the scaled parameters
    problem_parameters = importer.petab_problem.x_nominal_scaled

    # simulate
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)

    # extract results
    llh = -ret["fval"]
    simulation_df = objective_rr.simulation_to_measurement_df(
        ret["simulation_results"], petab_problem.measurement_df
    )

    simulation_df = simulation_df.rename(
        columns={petab.SIMULATION: petab.MEASUREMENT}
    )
    petab.check_measurement_df(simulation_df, petab_problem.observable_df)
    simulation_df = simulation_df.rename(
        columns={petab.MEASUREMENT: petab.SIMULATION}
    )
    simulation_df[petab.TIME] = simulation_df[petab.TIME].astype(int)

    # check if matches
    llhs_match = petabtests.evaluate_llh(llh, gt_llh, tol_llh)
    simulations_match = petabtests.evaluate_simulations(
        [simulation_df], gt_simulation_dfs, tol_simulations
    )

    # log matches
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"LLH: simulated: {llh}, expected: {gt_llh}, match = {llhs_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"Simulations: match = {simulations_match}",
    )

    if not all([llhs_match, simulations_match]):
        logger.error(f"Case {version}/{model_type}/{case} failed.")
        raise AssertionError(
            f"Case {case}: Test results do not match expectations"
        )

    logger.info(f"Case {version}/{model_type}/{case} passed.")


def test_deepcopy():
    """Test that deepcopy works as intended"""
    model_name = "Boehm_JProteomeRes2014"
    petab_problem = petab.Problem.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    petab_problem.model_name = model_name
    importer = objective_rr.PetabImporterRR(petab_problem)
    problem_parameters = petab_problem.x_nominal_free_scaled

    problem = importer.create_problem()
    obj = problem.objective

    problem_copied = copy.deepcopy(problem)
    copied_objective = problem_copied.objective

    assert obj(problem_parameters) == copied_objective(problem_parameters)

    # !!not adviced, only done here for testing purposes!!
    obj.roadrunner_instance.removeParameter(
        "pSTAT5A_rel", forceRegenerate=False
    )
    obj.roadrunner_instance.addParameter("pSTAT5A_rel", 0.0, False)
    obj.roadrunner_instance.addAssignmentRule(
        "pSTAT5A_rel", "(100 * pApB + 200 * pApA * specC17)"
    )

    assert obj(problem_parameters) != copied_objective(problem_parameters)


# write unit test to check whether roadrunner objective works with
# multiprocessing
def test_multiprocessing():
    """Test that multiprocessing works as intended"""
    model_name = "Boehm_JProteomeRes2014"
    petab_problem = petab.Problem.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    petab_problem.model_name = model_name
    importer = objective_rr.PetabImporterRR(petab_problem)

    problem = importer.create_problem()
    # start 30 times from the same point
    start_points = [problem.get_full_vector(problem.get_startpoints(1))] * 30
    problem.set_x_guesses(np.vstack(start_points))

    # for later comparisons, do one optimization run with single core
    result_single = pypesto.optimize.minimize(
        problem=problem,
        n_starts=1,
        engine=pypesto.engine.SingleCoreEngine(),
        progress_bar=False,
    )

    engine = pypesto.engine.MultiProcessEngine(n_procs=15)

    result = pypesto.optimize.minimize(
        problem=problem,
        n_starts=30,
        engine=engine,
        progress_bar=True,
    )
    assert np.all(
        [
            fval == result_single.optimize_result.fval[0]
            for fval in result.optimize_result.fval
        ]
    )
