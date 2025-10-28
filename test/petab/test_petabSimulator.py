"""Run PEtab tests for PetabSimulatorObjective."""

import logging

import basico.petab
import petab.v1 as petab
import petabtests
import pytest

from pypesto.objective.petab import PetabSimulatorObjective

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
    # FIXME https://github.com/ICB-DCM/pyPESTO/issues/1583
    if model_type == "sbml" and version == "v1.0.0" and case == "0020":
        pytest.xfail(reason="https://github.com/ICB-DCM/pyPESTO/issues/1583")

    try:
        _execute_case(case, model_type, version)
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


def _execute_case(case, model_type, version):
    """Run a single PEtab test suite case"""
    case = petabtests.test_id_str(case)
    logger.info(f"Case {case}")
    if case in ["0006", "0009", "0010", "0017", "0018", "0019"]:
        pytest.skip("Basico does not support these functionalities.")

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

    # import and create objective function
    petab_problem = petab.Problem.from_yaml(yaml_file)
    simulator = basico.petab.PetabSimulator(petab_problem)
    obj = PetabSimulatorObjective(simulator)

    # the scaled parameters
    problem_parameters = petab_problem.x_nominal_scaled

    # simulate
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)

    # extract results
    llh = -ret["fval"]
    simulation_df = ret["simulations"]

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
