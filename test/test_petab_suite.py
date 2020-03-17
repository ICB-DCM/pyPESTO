"""Execute petab test suite."""

import petabtests
import pypesto

import sys
import os
import pytest
from _pytest.outcomes import Skipped
import logging

try:
    import petab
    import amici.petab_objective
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_petab_suite():
    """Execute all cases from the petab test suite, report performance."""
    n_success = n_skipped = 0
    for case in petabtests.CASES_LIST:
        try:
            execute_case(case)
            n_success += 1
        except Skipped:
            n_skipped += 1
        except Exception as e:
            # run all despite failures
            logger.error(f"Case {case} failed.")
            logger.error(e)

    logger.info(f"{n_success} / {len(petabtests.CASES_LIST)} successful, "
                f"{n_skipped} skipped")
    if n_success + n_skipped != len(petabtests.CASES_LIST):
        sys.exit(1)


def execute_case(case):
    """Wrapper for _execute_case for handling test outcomes"""
    try:
        _execute_case(case)
    except Exception as e:
        if isinstance(e, NotImplementedError) \
                or "Timepoint-specific parameter overrides" in str(e):
            logger.info(
                f"Case {case} expectedly failed. Required functionality is "
                f"not implemented: {e}")
            pytest.skip(str(e))
        else:
            raise e


def _execute_case(case):
    """Run a single PEtab test suite case"""
    case = petabtests.test_id_str(case)
    logger.info(f"Case {case}")

    # case folder
    case_dir = os.path.join(petabtests.CASES_DIR, case)

    # load solution
    solution = petabtests.load_solution(case)
    gt_chi2 = solution[petabtests.CHI2]
    gt_llh = solution[petabtests.LLH]
    gt_simulation_dfs = solution[petabtests.SIMULATION_DFS]
    tol_chi2 = solution[petabtests.TOL_CHI2]
    tol_llh = solution[petabtests.TOL_LLH]
    tol_simulations = solution[petabtests.TOL_SIMULATIONS]

    # import petab problem
    yaml_file = os.path.join(case_dir, petabtests.problem_yaml_name(case))

    # unique folder for compiled amici model
    output_folder = f'amici_models/model_{case}'

    # import and create objective function
    importer = pypesto.PetabImporter.from_yaml(
        yaml_file, output_folder=output_folder)
    model = importer.create_model()
    obj = importer.create_objective(model=model)

    # the scaled parameters
    problem_parameters = importer.petab_problem.x_nominal_scaled

    # simulate
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)

    # extract results
    rdatas = ret['rdatas']
    chi2 = sum(rdata['chi2'] for rdata in rdatas)
    llh = - ret['fval']
    simulation_df = amici.petab_objective.rdatas_to_measurement_df(
        rdatas, model, importer.petab_problem.measurement_df)
    petab.check_measurement_df(
        simulation_df, importer.petab_problem.observable_df)
    simulation_df = simulation_df.rename(
        columns={petab.MEASUREMENT: petab.SIMULATION})
    simulation_df[petab.TIME] = simulation_df[petab.TIME].astype(int)

    # check if matches
    chi2s_match = petabtests.evaluate_chi2(chi2, gt_chi2, tol_chi2)
    llhs_match = petabtests.evaluate_llh(llh, gt_llh, tol_llh)
    simulations_match = petabtests.evaluate_simulations(
        [simulation_df], gt_simulation_dfs, tol_simulations)

    # log matches
    logger.log(logging.INFO if chi2s_match else logging.ERROR,
               f"CHI2: simulated: {chi2}, expected: {gt_chi2},"
               f" match = {chi2s_match}")
    logger.log(logging.INFO if simulations_match else logging.ERROR,
               f"LLH: simulated: {llh}, expected: {gt_llh}, "
               f"match = {llhs_match}")
    logger.log(logging.INFO if simulations_match else logging.ERROR,
               f"Simulations: match = {simulations_match}")

    # FIXME case 7, 16 fail due to amici/#963
    if not all([llhs_match, simulations_match]) \
            or (not chi2s_match and case not in ['0007', '0016']):
        # chi2s_match ignored until fixed in amici
        logger.error(f"Case {case} failed.")
        raise AssertionError(f"Case {case}: Test results do not match "
                             "expectations")

    logger.info(f"Case {case} passed.")
