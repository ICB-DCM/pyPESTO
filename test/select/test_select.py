from functools import partial
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import petab_select
import pytest
from more_itertools import one
from petab_select import ESTIMATE, Criterion, Method, Model

import pypesto.select
import pypesto.visualize.select
from pypesto.select import model_problem

# Options sent to `pypesto.optimize.optimize.minimize`, to reduce run time.
minimize_options = {
    'n_starts': 10,
}
# Tolerances for the differences between expected and test values.
tolerances = {
    'rtol': 1e-2,
    'atol': 1e-2,
}


@pytest.fixture
def petab_problem_yaml() -> Path:
    """The location of the PEtab problem YAML file."""
    return (
        Path(__file__).parent.parent.parent
        / 'doc'
        / 'example'
        / 'model_selection'
        / 'example_modelSelection.yaml'
    )


@pytest.fixture
def petab_select_problem_yaml() -> Path:
    """The location of the PEtab Select problem YAML file."""
    return (
        Path(__file__).parent.parent.parent
        / 'doc'
        / 'example'
        / 'model_selection'
        / 'petab_select_problem.yaml'
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
def initial_models(petab_problem_yaml) -> List[Model]:
    """Models that can be used to initialize a search."""
    initial_model_1 = Model(
        model_id='myModel1',
        petab_yaml=petab_problem_yaml,
        parameters={
            'k1': 0,
            'k2': 0,
            'k3': 0,
        },
        criteria={Criterion.AIC: np.inf},
    )
    initial_model_2 = Model(
        model_id='myModel2',
        petab_yaml=petab_problem_yaml,
        parameters={
            'k1': ESTIMATE,
            'k2': ESTIMATE,
            'k3': 0,
        },
        criteria={Criterion.AIC: np.inf},
    )
    initial_models = [initial_model_1, initial_model_2]
    return initial_models


def test_problem_select(pypesto_select_problem):
    """Test the `Problem.select` method."""
    # Iteration 1 #############################################################
    best_model_1, local_history_1, _ = pypesto_select_problem.select(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        minimize_options=minimize_options,
    )

    expected_local_history_model_subspace_ids = ['M1_0']
    test_local_history_model_subspace_ids = [
        model.model_subspace_id for model in local_history_1.values()
    ]
    # The expected "forward" models were found.
    assert (
        test_local_history_model_subspace_ids
        == expected_local_history_model_subspace_ids
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

    # Iteration 2 #############################################################
    best_model_2, local_history_2, history_2 = pypesto_select_problem.select(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        minimize_options=minimize_options,
    )

    expected_local_history_model_subspace_ids = ['M1_1', 'M1_2', 'M1_3']
    test_local_history_model_subspace_ids = [
        model.model_subspace_id for model in local_history_2.values()
    ]
    # The expected "forward" models were found.
    assert (
        test_local_history_model_subspace_ids
        == expected_local_history_model_subspace_ids
    )

    expected_best_model_subspace_id = 'M1_3'
    test_best_model_subspace_id = best_model_2.model_subspace_id
    # The best model is as expected.
    assert test_best_model_subspace_id == expected_best_model_subspace_id

    expected_best_model_aic = -4.71
    test_best_model_aic = best_model_2.get_criterion(Criterion.AIC)
    # The best model has its criterion value set and is the
    # expected value.
    assert np.isclose(
        [test_best_model_aic],
        [expected_best_model_aic],
        **tolerances,
    )


def test_problem_select_to_completion(pypesto_select_problem):
    """Test the `Problem.select_to_completion` method."""
    best_models = pypesto_select_problem.select_to_completion(
        method=Method.FORWARD,
        criterion=Criterion.BIC,
        select_first_improvement=True,
        startpoint_latest_mle=True,
        minimize_options=minimize_options,
    )

    expected_history_subspace_ids = ['M1_0', 'M1_1', 'M1_4', 'M1_5', 'M1_7']
    test_history_subspace_ids = [
        model.model_subspace_id
        for model in pypesto_select_problem.history.values()
    ]
    # Expected models were calibrated during the search.
    assert test_history_subspace_ids == expected_history_subspace_ids

    expected_best_model_subspace_ids = ['M1_0', 'M1_1', 'M1_7']
    test_best_model_subspace_ids = [
        model.model_subspace_id for model in best_models
    ]
    # Expected best models were found.
    assert test_best_model_subspace_ids == expected_best_model_subspace_ids

    expected_best_model_criterion_values = [36.767, -4.592, -4.889]
    test_best_model_criterion_values = [
        model.get_criterion(Criterion.BIC) for model in best_models
    ]
    # The best models have the expected criterion values.
    assert np.isclose(
        test_best_model_criterion_values,
        expected_best_model_criterion_values,
        **tolerances,
    ).all()


def test_problem_multistart_select(pypesto_select_problem, initial_models):
    """Test the `Problem.multistart_select` method."""
    best_model, best_models = pypesto_select_problem.multistart_select(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        predecessor_models=initial_models,
        minimize_options=minimize_options,
    )

    expected_best_model_subspace_id = 'M1_3'
    test_best_model_subspace_id = best_model.model_subspace_id
    # The best model is as expected.
    assert test_best_model_subspace_id == expected_best_model_subspace_id

    expected_best_models_criterion_values = {
        'M1_3': -4.705,
        # 'M1_7': -4.056,  # skipped -- reproducibility requires many starts
    }
    test_best_models_criterion_values = {
        model.model_subspace_id: model.get_criterion(Criterion.AIC)
        for model in best_models
        if model.model_subspace_id != 'M1_7'  # skipped, see above
    }
    # The best models are as expected and have the expected criterion values.
    pd.testing.assert_series_equal(
        pd.Series(test_best_models_criterion_values),
        pd.Series(expected_best_models_criterion_values),
        **tolerances,
    )

    initial_model_id_hash_map = {
        initial_model.model_id: initial_model.get_hash()
        for initial_model in initial_models
    }
    expected_predecessor_model_hashes = {
        'M1_1': initial_model_id_hash_map['myModel1'],
        'M1_2': initial_model_id_hash_map['myModel1'],
        'M1_3': initial_model_id_hash_map['myModel1'],
        'M1_7': initial_model_id_hash_map['myModel2'],
    }
    test_predecessor_model_hashes = {
        model.model_subspace_id: model.predecessor_model_hash
        for model in pypesto_select_problem.history.values()
    }
    # All calibrated models have the expected predecessor model.
    assert test_predecessor_model_hashes == expected_predecessor_model_hashes


def test_postprocessors(petab_select_problem):
    """Test model calibration postprocessors."""
    output_path = Path('output')
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
    pypesto_select_problem = pypesto.select.Problem(
        petab_select_problem=petab_select_problem,
        model_postprocessor=multi_postprocessor,
    )

    # Iteration 1 # Same as first iteration of `test_problem_select` ##########
    best_model_1, local_history_1, _ = pypesto_select_problem.select(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        minimize_options=minimize_options,
    )

    expected_local_history_model_subspace_ids = ['M1_0']
    test_local_history_model_subspace_ids = [
        model.model_subspace_id for model in local_history_1.values()
    ]
    # The expected "forward" models were found.
    assert (
        test_local_history_model_subspace_ids
        == expected_local_history_model_subspace_ids
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

    expected_png_file = output_path / (best_model_1.model_hash + ".png")
    expected_hdf5_file = output_path / (best_model_1.model_hash + ".hdf5")

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


def test_vis(pypesto_select_problem):
    """Test plotting routines."""
    best_models = pypesto_select_problem.select_to_completion(
        method=Method.FORWARD,
        criterion=Criterion.AIC,
        minimize_options=minimize_options,
    )
    pypesto.visualize.select.plot_selected_models(
        selected_models=best_models,
        criterion=Criterion.AIC,
    )
    pypesto.visualize.select.plot_history_digraph(
        problem=pypesto_select_problem,
        criterion=Criterion.AIC,
    )
