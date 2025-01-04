from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import petab.v1 as petab
import petab_select
import pytest
from more_itertools import one
from petab.v1.C import NOMINAL_VALUE
from petab_select import (
    ESTIMATE,
    PETAB_PROBLEM,
    VIRTUAL_INITIAL_MODEL,
    Criterion,
    Method,
    Model,
    get_best_by_iteration,
)

import pypesto.engine
import pypesto.select
import pypesto.visualize.select
from pypesto.select import SacessMinimizeMethod, model_problem
from pypesto.select.misc import correct_x_guesses

model_problem_options = {
    # Options sent to `pypesto.optimize.optimize.minimize`, to reduce run time.
    "minimize_options": {
        "engine": pypesto.engine.MultiProcessEngine(),
        "n_starts": 20,
        "filename": None,
        "progress_bar": False,
    }
}
# Tolerances for the differences between expected and test values.
tolerances = {
    "rtol": 1e-2,
    "atol": 1e-2,
}


@pytest.fixture
def petab_problem_yaml() -> Path:
    """The location of the PEtab problem YAML file."""
    return (
        Path(__file__).parent.parent.parent
        / "doc"
        / "example"
        / "model_selection"
        / "example_modelSelection.yaml"
    )


@pytest.fixture
def petab_select_problem_yaml() -> Path:
    """The location of the PEtab Select problem YAML file."""
    return (
        Path(__file__).parent.parent.parent
        / "doc"
        / "example"
        / "model_selection"
        / "petab_select_problem.yaml"
    )


@pytest.fixture
def petab_select_problem(petab_select_problem_yaml) -> petab_select.Problem:
    """The PEtab Select problem."""
    return petab_select.Problem.from_yaml(petab_select_problem_yaml)


@pytest.fixture
def pypesto_select_problem(petab_select_problem) -> pypesto.select.Problem:
    """The pyPESTO model selection problem."""
    return pypesto.select.Problem(petab_select_problem=petab_select_problem)


@pytest.fixture
def initial_models(petab_problem_yaml) -> list[Model]:
    """Models that can be used to initialize a search."""
    initial_model_1 = Model(
        model_id="myModel1",
        model_subspace_id="dummy",
        model_subspace_indices=[],
        model_subspace_petab_yaml=petab_problem_yaml,
        parameters={
            "k1": 0,
            "k2": 0,
            "k3": 0,
        },
        criteria={Criterion.AIC: np.inf},
    )
    initial_model_2 = Model(
        model_id="myModel2",
        model_subspace_id="dummy",
        model_subspace_indices=[],
        model_subspace_petab_yaml=petab_problem_yaml,
        parameters={
            "k1": ESTIMATE,
            "k2": ESTIMATE,
            "k3": 0,
        },
        criteria={Criterion.AIC: np.inf},
    )
    initial_models = [initial_model_1, initial_model_2]
    return initial_models


def test_problem_select(pypesto_select_problem):
    """Test the `Problem.select` method."""
    expected_results = [
        {
            "candidates_model_subspace_ids": ["M1_0"],
            "best_model_subspace_id": "M1_0",
            "best_model_aic": 36.97,
        },
        {
            "candidates_model_subspace_ids": ["M1_1", "M1_2", "M1_3"],
            "best_model_subspace_id": "M1_3",
            "best_model_aic": -4.71,
        },
        {
            "candidates_model_subspace_ids": ["M1_5", "M1_6"],
            "best_model_subspace_id": "M1_6",
            "best_model_aic": -4.15,
        },
        {
            "candidates_model_subspace_ids": ["M1_7"],
            "best_model_subspace_id": "M1_7",
            "best_model_aic": -4.06,
        },
    ]

    candidate_space = (
        pypesto_select_problem.petab_select_problem.new_candidate_space(
            method=Method.FORWARD
        )
    )
    criterion = Criterion.AIC

    best_model = None
    for expected_result in expected_results:
        best_model, _ = pypesto_select_problem.select(
            criterion=criterion,
            model_problem_options=model_problem_options,
            predecessor_model=best_model,
            candidate_space=candidate_space,
        )

        test_candidates_model_subspace_ids = [
            model.model_subspace_id for model in candidate_space.models
        ]

        test_best_model_subspace_id = best_model.model_subspace_id
        test_best_model_aic = best_model.get_criterion(Criterion.AIC)

        test_result = {
            "candidates_model_subspace_ids": test_candidates_model_subspace_ids,
            "best_model_subspace_id": test_best_model_subspace_id,
            "best_model_aic": test_best_model_aic,
        }

        # The expected "forward" models were found.
        assert (
            test_result["candidates_model_subspace_ids"]
            == expected_result["candidates_model_subspace_ids"]
        )

        # The best model is as expected.
        assert (
            test_result["best_model_subspace_id"]
            == expected_result["best_model_subspace_id"]
        )

        # The best model has its criterion value set and is the expected value.
        assert np.isclose(
            [test_result["best_model_aic"]],
            [expected_result["best_model_aic"]],
            **tolerances,
        )


def test_problem_select_to_completion(pypesto_select_problem):
    """Test the `Problem.select_to_completion` method."""
    candidate_space = (
        pypesto_select_problem.petab_select_problem.new_candidate_space(
            method=Method.FORWARD
        )
    )
    models = pypesto_select_problem.select_to_completion(
        criterion=Criterion.BIC,
        select_first_improvement=True,
        startpoint_latest_mle=True,
        model_problem_options=model_problem_options,
        candidate_space=candidate_space,
    )

    expected_calibrated_ids = {
        "M1_0",
        "M1_1",
        "M1_4",
        "M1_5",
        "M1_7",
    }
    test_calibrated_ids = {model.model_subspace_id for model in models}
    # Expected models were calibrated during the search.
    assert test_calibrated_ids == expected_calibrated_ids

    best_by_iteration = get_best_by_iteration(
        models=models, criterion=Criterion.BIC
    )

    expected_best_by_iteration = [
        "M1_0",
        "M1_1",
        "M1_5",
        "M1_7",
    ]
    test_best_by_iteration = [
        model.model_subspace_id for model in best_by_iteration.values()
    ]
    # Expected best models were found.
    assert test_best_by_iteration == expected_best_by_iteration

    expected_best_values = [
        36.767,
        -4.592,
        # This iteration with models `{'M1_4', 'M1_5'}` didn't have a better
        # model than the previous iteration.
        -3.330,
        -4.889,
    ]
    test_best_values = [
        (
            model.get_criterion(Criterion.BIC)
            if model != VIRTUAL_INITIAL_MODEL
            else np.inf
        )
        for model in best_by_iteration.values()
    ]
    # The best models have the expected criterion values.
    assert np.isclose(
        test_best_values,
        expected_best_values,
        **tolerances,
    ).all()


def test_problem_multistart_select(pypesto_select_problem, initial_models):
    """Test the `Problem.multistart_select` method."""
    criterion = Criterion.AIC
    best_model, model_lists = pypesto_select_problem.multistart_select(
        method=Method.FORWARD,
        criterion=criterion,
        predecessor_models=initial_models,
        model_problem_options=model_problem_options,
    )

    expected_best_model_subspace_id = "M1_3"
    test_best_model_subspace_id = best_model.model_subspace_id
    # The best model is as expected.
    assert test_best_model_subspace_id == expected_best_model_subspace_id

    best_models = [
        petab_select.ui.get_best(
            problem=pypesto_select_problem.petab_select_problem,
            models=models,
            criterion=criterion,
        )
        for models in model_lists
    ]

    expected_best_models_criterion_values = {
        "M1_3": -4.705,
        # 'M1_7': -4.056,  # skipped -- reproducibility requires many starts
    }
    # As M1_7 criterion comparison is skipped, at least ensure it is present
    assert {m.model_subspace_id for m in best_models} == {"M1_3", "M1_7"}
    test_best_models_criterion_values = {
        model.model_subspace_id: model.get_criterion(Criterion.AIC)
        for model in best_models
        if model.model_subspace_id != "M1_7"  # skipped, see above
    }
    # The best models are as expected and have the expected criterion values.
    pd.testing.assert_series_equal(
        pd.Series(test_best_models_criterion_values),
        pd.Series(expected_best_models_criterion_values),
        **tolerances,
    )

    initial_model_id_hash_map = {
        initial_model.model_id: initial_model.hash
        for initial_model in initial_models
    }
    expected_predecessor_model_hashes = {
        "M1_1": initial_model_id_hash_map["myModel1"],
        "M1_2": initial_model_id_hash_map["myModel1"],
        "M1_3": initial_model_id_hash_map["myModel1"],
        "M1_7": initial_model_id_hash_map["myModel2"],
    }
    test_predecessor_model_hashes = {
        model.model_subspace_id: model.predecessor_model_hash
        for model in pypesto_select_problem.calibrated_models
    }
    # All calibrated models have the expected predecessor model.
    assert test_predecessor_model_hashes == expected_predecessor_model_hashes


def test_postprocessors(petab_select_problem):
    """Test model calibration postprocessors."""
    output_path = Path("output")
    output_path.mkdir(exist_ok=True, parents=True)
    postprocessor_1 = partial(
        pypesto.select.postprocessors.save_postprocessor,
        output_path=output_path,
    )
    postprocessor_2 = partial(
        pypesto.select.postprocessors.waterfall_plot_postprocessor,
        output_path=output_path,
    )
    multi_postprocessor = partial(
        pypesto.select.postprocessors.multi_postprocessor,
        postprocessors=[postprocessor_1, postprocessor_2],
    )
    model_problem_options = {
        "postprocessor": multi_postprocessor,
    }
    pypesto_select_problem = pypesto.select.Problem(
        petab_select_problem=petab_select_problem,
        model_problem_options=model_problem_options,
    )

    # Iteration 1 # Same as first iteration of `test_problem_select` ##########
    best_model_1, newly_calibrated_models_1 = pypesto_select_problem.select(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        model_problem_options=model_problem_options,
    )

    expected_newly_calibrated_models_subspace_ids = ["M1_0"]
    test_newly_calibrated_models_subspace_ids = [
        model.model_subspace_id for model in newly_calibrated_models_1
    ]
    # The expected "forward" models were found.
    assert (
        test_newly_calibrated_models_subspace_ids
        == expected_newly_calibrated_models_subspace_ids
    )

    expected_best_model_aic = 36.97
    test_best_model_aic = best_model_1.get_criterion(Criterion.AIC)
    # The best model (only model) has its criterion value set and is the
    # expected value.
    assert np.isclose(
        [test_best_model_aic],
        [expected_best_model_aic],
        **tolerances,
    )
    # End Iteration 1 #########################################################

    expected_png_file = output_path / f"{best_model_1.hash}.png"
    expected_hdf5_file = output_path / f"{best_model_1.hash}.hdf5"

    # The expected files exist.
    assert expected_png_file.is_file()
    assert expected_hdf5_file.is_file()

    # Remove the expected files (also ensures they firstly exist).
    expected_png_file.unlink()
    expected_hdf5_file.unlink()


def test_model_problem_fake_result():
    """Test fake results for models with no estimated parameters."""
    expected_fval = 100.0

    fake_result = model_problem.create_fake_pypesto_result_from_fval(
        fval=expected_fval,
    )
    # There is only one start in the result.
    fake_start = one(fake_result.optimize_result.list)

    expected_id = "fake_result_for_problem_with_no_estimated_parameters"
    test_id = fake_start.id
    # The fake start has the expected fake ID.
    assert test_id == expected_id

    expected_x = []
    test_x = fake_start.x.tolist()
    # The fake start has the expected zero estimated parameters.
    assert test_x == expected_x

    test_fval = fake_start.fval
    # The fake start has the expected fval.
    assert test_fval == expected_fval


def test_custom_objective(petab_problem_yaml):
    parameters = {
        "k2": 0.333,
        "sigma_x2": 0.444,
    }
    parameters_x_guesses = [parameters]

    expected_fun = 0.111
    expected_grad = [0.222] * 3

    def fun_grad(x, fun=expected_fun, grad=expected_grad):
        return expected_fun, expected_grad

    custom_objective = pypesto.objective.Objective(
        fun=fun_grad,
        grad=True,
    )

    model = Model(
        model_subspace_id="dummy",
        model_subspace_indices=[],
        model_subspace_petab_yaml=petab_problem_yaml,
        parameters={},
    )

    def get_pypesto_problem(model, objective, petab_problem, x_guesses):
        corrected_x_guesses = correct_x_guesses(
            x_guesses=x_guesses,
            model=model,
        )

        pypesto_problem = pypesto.Problem(
            objective=objective,
            lb=petab_problem.lb,
            ub=petab_problem.ub,
            x_guesses=corrected_x_guesses,
        )

        return pypesto_problem

    def model_to_pypesto_problem(
        model, objective=custom_objective, x_guesses=None
    ):
        petab_problem = petab_select.ui.model_to_petab(model=model)[
            PETAB_PROBLEM
        ]
        pypesto_problem = get_pypesto_problem(
            model=model,
            objective=objective,
            petab_problem=petab_problem,
            x_guesses=x_guesses,
        )
        return pypesto_problem

    pypesto_problem = model_to_pypesto_problem(
        model, x_guesses=parameters_x_guesses
    )

    expected_x_guess = petab.Problem.from_yaml(
        petab_problem_yaml
    ).parameter_df[NOMINAL_VALUE]
    expected_x_guess.update(parameters)
    expected_x_guess = expected_x_guess.values

    test_x_guess = pypesto_problem.x_guesses[0]
    # The x_guess was generated correctly from the partial dictionary, with remaining results taken from the nominal values of the PEtab parameter table.
    assert np.isclose(test_x_guess, expected_x_guess).all()

    test_fun, test_grad = pypesto_problem.objective(
        test_x_guess, sensi_orders=(0, 1)
    )
    # The custom objective and gradient were returned.
    assert test_fun == expected_fun
    assert np.isclose(test_grad, expected_grad).all()


def test_sacess_minimize_method(pypesto_select_problem, initial_models):
    """Test `SacessMinimizeMethod`.

    Only ensures that the pipeline runs.
    """
    predecessor_model = initial_models[1]

    minimize_method = SacessMinimizeMethod(
        num_workers=2,
        max_walltime_s=1,
    )

    model_problem_options = {
        "minimize_method": minimize_method,
    }

    pypesto_select_problem.select_to_completion(
        model_problem_options=model_problem_options,
        method=petab_select.Method.FORWARD,
        predecessor_model=predecessor_model,
    )
