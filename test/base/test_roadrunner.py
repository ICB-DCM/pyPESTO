"""Test the roadrunner interface."""

import copy
import logging
import os

import benchmark_models_petab as models
import numpy as np
import pandas as pd
import petab.v1 as petab
import petabtests
import pytest

import pypesto
import pypesto.petab
from pypesto.objective.roadrunner import simulation_to_measurement_df
from pypesto.objective.roadrunner.utils import inject_timepoint_specific_noise

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
        if isinstance(e, NotImplementedError):
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
    if case == "0018" and model_type == "sbml" and version == "v1.0.0":
        pytest.skip("https://github.com/ICB-DCM/pyPESTO/issues/1597")
    if case == "0006" and model_type == "sbml" and version == "v1.0.0":
        pytest.skip(
            "Case 0006: Timepoint-specific observable parameters not yet supported for roadrunner"
        )
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

    importer = pypesto.petab.PetabImporter.from_yaml(
        yaml_file, simulator_type="roadrunner"
    )
    petab_problem = importer.petab_problem
    obj = importer.create_problem().objective

    # the scaled parameters
    problem_parameters = importer.petab_problem.x_nominal_free_scaled

    # simulate
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)

    # extract results
    llh = -ret["fval"]
    simulation_df = simulation_to_measurement_df(
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
    importer = pypesto.petab.PetabImporter(
        petab_problem, simulator_type="roadrunner"
    )
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


def test_multiprocessing():
    """Test that multiprocessing works as intended"""
    model_name = "Boehm_JProteomeRes2014"
    petab_problem = petab.Problem.from_yaml(
        os.path.join(models.MODELS_DIR, model_name, model_name + ".yaml")
    )
    petab_problem.model_name = model_name
    importer = pypesto.petab.PetabImporter(
        petab_problem, simulator_type="roadrunner"
    )

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

    engine = pypesto.engine.MultiProcessEngine(n_procs=8)

    result = pypesto.optimize.minimize(
        problem=problem,
        n_starts=15,
        engine=engine,
        progress_bar=True,
    )
    assert np.all(
        [
            fval == result_single.optimize_result.fval[0]
            for fval in result.optimize_result.fval
        ]
    )


def test_inject_timepoint_specific_noise_no_truncation():
    """Timepoint-specific numeric noise overrides must not be silently
    truncated by a fixed-width numpy string dtype inferred from the
    (possibly short) default noise formula, e.g. "1"."""
    noise_distributions = np.array(["lin_normal"])
    noise_formulae = np.array(["1"])
    observable_ids = ["obs_a"]
    measurement_df = pd.DataFrame(
        {
            "observableId": ["obs_a", "obs_a"],
            "time": [0, 10],
            "measurement": [0.7, 0.1],
            "noiseParameters": [5.0, 25.0],
        }
    )
    measurements = np.array([[0, 0.7], [10, 0.1]])

    _, noise_formulae_out = inject_timepoint_specific_noise(
        noise_distributions,
        noise_formulae,
        measurement_df,
        observable_ids,
        measurements,
    )

    assert [float(v) for v in noise_formulae_out.ravel()] == [5.0, 25.0]


def test_inject_timepoint_specific_noise_replicates():
    """Replicate measurements at the same timepoint with different noise
    overrides must keep their individual values, not collapse to a single
    value shared by all replicates at that timepoint."""
    noise_distributions = np.array(["lin_normal"])
    noise_formulae = np.array(["1.0"])
    observable_ids = ["obs_a"]
    measurement_df = pd.DataFrame(
        {
            "observableId": ["obs_a", "obs_a", "obs_a"],
            "time": [0, 0, 10],
            "measurement": [0.7, 0.75, 0.1],
            "noiseParameters": [3.0, 9.0, 3.0],
        }
    )
    measurements = np.array([[0, 0.7], [0, 0.75], [10, 0.1]])

    _, noise_formulae_out = inject_timepoint_specific_noise(
        noise_distributions,
        noise_formulae,
        measurement_df,
        observable_ids,
        measurements,
    )

    assert [float(v) for v in noise_formulae_out.ravel()] == [3.0, 9.0, 3.0]


def test_timepoint_specific_noise_2d_with_symbolic_placeholder():
    """When one observable's noise varies by timepoint (forcing the
    per-condition noise array to 2D), another observable in the same
    condition that still uses a *compound* noise formula referencing
    ``noiseParameter1_x``/``noiseParameter2_x`` placeholders (constant
    across timepoints, PEtab test case 0014 style) must still get that
    formula registered as a ``noiseFormula_x`` roadrunner parameter -- a
    bare placeholder name would instead resolve directly via the standard
    PEtab parameter mapping, without ever reaching this registration code.

    This exercises the ``noise_formulae_array.ndim == 2`` branch of
    ``RoadRunnerObjectiveCreator._check_noise_formulae``, which the
    purely-numeric test above never reaches.
    """
    from petab.v1.C import (
        CONDITION_ID,
        ESTIMATE,
        LIN,
        LOWER_BOUND,
        MEASUREMENT,
        NOISE_FORMULA,
        NOISE_PARAMETERS,
        NOMINAL_VALUE,
        OBSERVABLE_FORMULA,
        OBSERVABLE_ID,
        PARAMETER_ID,
        PARAMETER_SCALE,
        SIMULATION,
        SIMULATION_CONDITION_ID,
        TIME,
        UPPER_BOUND,
    )
    from petab.v1.calculate import calculate_llh
    from petab.v1.models.sbml_model import SbmlModel
    from petabtests.C import DEFAULT_SBML_FILE
    from petabtests.model import analytical_a, analytical_b

    a0, b0, k1, k2 = 1, 0, 0.8, 0.6
    times = [0, 10]

    condition_df = pd.DataFrame({CONDITION_ID: ["c0"]}).set_index(CONDITION_ID)
    measurement_df = pd.DataFrame(
        {
            OBSERVABLE_ID: ["obs_a", "obs_a", "obs_b", "obs_b"],
            SIMULATION_CONDITION_ID: ["c0"] * 4,
            TIME: times * 2,
            MEASUREMENT: [
                analytical_a(t, a0, b0, k1, k2) + 0.01 for t in times
            ]
            + [analytical_b(t, a0, b0, k1, k2) + 0.01 for t in times],
            # obs_a varies numerically by timepoint -> forces the array 2D.
            # obs_b uses the same compound override at both timepoints ->
            # stays an unresolved formula in that 2D array (matches PEtab
            # test case 0014's "0.5;2" pattern).
            NOISE_PARAMETERS: [0.2, 2.0, "0.1;0.2", "0.1;0.2"],
        }
    )
    observable_df = pd.DataFrame(
        {
            OBSERVABLE_ID: ["obs_a", "obs_b"],
            OBSERVABLE_FORMULA: ["A", "B"],
            NOISE_FORMULA: [
                "noiseParameter1_obs_a",
                "noiseParameter1_obs_b + noiseParameter2_obs_b",
            ],
        }
    ).set_index(OBSERVABLE_ID)
    parameter_df = pd.DataFrame(
        {
            PARAMETER_ID: ["a0", "b0", "k1", "k2"],
            PARAMETER_SCALE: [LIN] * 4,
            LOWER_BOUND: [0] * 4,
            UPPER_BOUND: [10] * 4,
            NOMINAL_VALUE: [a0, b0, k1, k2],
            ESTIMATE: [1] * 4,
        }
    ).set_index(PARAMETER_ID)

    petab_problem = petab.Problem(
        model=SbmlModel.from_file(DEFAULT_SBML_FILE),
        condition_df=condition_df,
        measurement_df=measurement_df,
        observable_df=observable_df,
        parameter_df=parameter_df,
    )

    simulation_df = measurement_df.rename(columns={MEASUREMENT: SIMULATION})
    simulation_df[SIMULATION] = [
        analytical_a(t, a0, b0, k1, k2) for t in times
    ] + [analytical_b(t, a0, b0, k1, k2) for t in times]
    expected_llh = calculate_llh(
        [measurement_df], [simulation_df], [observable_df], parameter_df
    )

    importer = pypesto.petab.PetabImporter(
        petab_problem, simulator_type="roadrunner"
    )
    obj = importer.create_problem().objective

    # the placeholder must have been registered as a roadrunner parameter
    assert obj.roadrunner_instance.getValue("noiseFormula_obs_b") is not None

    problem_parameters = petab_problem.x_nominal_free_scaled
    ret = obj(problem_parameters, sensi_orders=(0,), return_dict=True)
    llh = -ret["fval"]

    assert llh == pytest.approx(expected_llh, rel=1e-6)
