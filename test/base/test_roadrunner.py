"""Test the roadrunner interface."""

import logging

import petab
import petabtests
import pytest

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
    if case == "0005":
        print("Case 0005:")
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
    simulation_df = ret["simulation_results"]

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
        f"LLH: simulated: {llh}, expected: {gt_llh}, " f"match = {llhs_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"Simulations: match = {simulations_match}",
    )

    if not all([llhs_match, simulations_match]):
        logger.error(f"Case {version}/{model_type}/{case} failed.")
        raise AssertionError(
            f"Case {case}: Test results do not match " "expectations"
        )

    logger.info(f"Case {version}/{model_type}/{case} passed.")
