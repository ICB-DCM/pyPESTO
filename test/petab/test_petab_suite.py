"""Execute petab test suite."""

import logging

import amici.petab.simulations
import petab.v1 as petab
import petabtests
import pytest

import pypesto
import pypesto.petab

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    "case, model_type, version",
    [
        (case, model_type, version)
        for (model_type, version) in (
            ("sbml", "v1.0.0"),
            ("sbml", "v2.0.0"),
            ("pysb", "v2.0.0"),
        )
        for case in petabtests.get_cases(format_=model_type, version=version)
    ],
)
def test_petab_case(case, model_type, version):
    """Wrapper for _execute_case for handling test outcomes"""
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

    # case folder
    case_dir = petabtests.get_case_dir(case, model_type, version)

    # load solution
    solution = petabtests.load_solution(
        case, format=model_type, version=version
    )
    gt_chi2 = solution[petabtests.CHI2]
    gt_llh = solution[petabtests.LLH]
    gt_simulation_dfs = solution[petabtests.SIMULATION_DFS]
    tol_chi2 = solution[petabtests.TOL_CHI2]
    tol_llh = solution[petabtests.TOL_LLH]
    tol_simulations = solution[petabtests.TOL_SIMULATIONS]

    # import petab problem
    yaml_file = case_dir / petabtests.problem_yaml_name(case)

    # unique folder for compiled amici model
    model_name = (
        f"petab_test_case_{case}_{model_type}_{version.replace('.', '_')}"
    )

    # import and create objective function
    if case.startswith("0006"):
        if version == "v2.0.0":
            pytest.skip("TODO")
        petab_problem = petab.Problem.from_yaml(yaml_file)
        petab.flatten_timepoint_specific_output_overrides(petab_problem)
        importer = pypesto.petab.PetabImporter(
            petab_problem=petab_problem,
            model_name=model_name,
        )
        petab_problem = petab.Problem.from_yaml(yaml_file)
    else:
        importer = pypesto.petab.PetabImporter.from_yaml(
            yaml_file, model_name=model_name
        )
        petab_problem = importer.petab_problem

    factory = importer.create_objective_creator()
    model = factory.create_model(generate_sensitivity_code=False)
    obj = factory.create_objective(model=model)

    if version == "v1.0.0":
        # the scaled parameters
        problem_parameters = factory.petab_problem.x_nominal_scaled
    else:
        problem_parameters = importer.petab_problem.x_nominal

    # simulate
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)

    # extract results
    rdatas = ret["rdatas"]
    chi2 = sum(rdata["chi2"] for rdata in rdatas)
    llh = -ret["fval"]

    if version == "v1.0.0":
        simulation_df = amici.petab.simulations.rdatas_to_measurement_df(
            rdatas, model, importer.petab_problem.measurement_df
        )

        if case.startswith("0006"):
            simulation_df = petab.unflatten_simulation_df(
                simulation_df, petab_problem
            )

        petab.check_measurement_df(simulation_df, petab_problem.observable_df)
        simulation_df = simulation_df.rename(
            columns={petab.MEASUREMENT: petab.SIMULATION}
        )
        simulation_df[petab.TIME] = simulation_df[petab.TIME].astype(int)
    else:
        simulation_df = obj.rdatas_to_simulation_df(rdatas)

    # check if matches
    chi2s_match = petabtests.evaluate_chi2(chi2, gt_chi2, tol_chi2)
    llhs_match = petabtests.evaluate_llh(llh, gt_llh, tol_llh)
    simulations_match = petabtests.evaluate_simulations(
        [simulation_df], gt_simulation_dfs, tol_simulations
    )

    # log matches
    logger.log(
        logging.INFO if chi2s_match else logging.ERROR,
        f"CHI2: simulated: {chi2}, expected: {gt_chi2}, match = {chi2s_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"LLH: simulated: {llh}, expected: {gt_llh}, match = {llhs_match}",
    )
    logger.log(
        logging.INFO if simulations_match else logging.ERROR,
        f"Simulations: match = {simulations_match}",
    )

    if not all([llhs_match, chi2s_match, simulations_match]):
        logger.error(f"Case {version}/{model_type}/{case} failed.")
        raise AssertionError(
            f"Case {case}: Test results do not match expectations"
        )

    logger.info(f"Case {version}/{model_type}/{case} passed.")
