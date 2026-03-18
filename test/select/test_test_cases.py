import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import petab_select
import pytest
import yaml
from petab_select import Model
from petab_select.constants import (
    CRITERIA,
    ESTIMATED_PARAMETERS,
    TERMINATE,
)

import pypesto.engine
import pypesto.optimize
import pypesto.select

# Set to `[]` to test all
test_cases = [
    # "0009",
]

# Do not use computationally-expensive test cases in CI
skip_test_cases = [
    "0009",
]

# Download test cases from GitHub
test_cases_path = Path("petab_select") / "test_cases"
if not test_cases_path.exists():
    subprocess.run([Path(__file__).parent / "get_test_cases.sh"])  # noqa: S603

# Reduce runtime but with high reproducibility
minimize_options = {
    "n_starts": 10,
    "engine": pypesto.engine.MultiProcessEngine(),
    "filename": None,
    "progress_bar": False,
}


def objective_customizer(obj):
    obj.amici_solver.set_relative_tolerance(1e-12)


model_problem_options = {
    "minimize_options": minimize_options,
    "objective_customizer": objective_customizer,
}


@pytest.mark.parametrize(
    "test_case_path_stem",
    sorted(
        [test_case_path.stem for test_case_path in test_cases_path.glob("*")]
    ),
)
def test_pypesto(test_case_path_stem):
    """Run all test cases with pyPESTO."""
    if test_cases and test_case_path_stem not in test_cases:
        pytest.skip("Test excluded from subset selected for debugging.")
        return

    if test_case_path_stem in skip_test_cases:
        pytest.skip("Test marked to be skipped.")
        return

    test_case_path = test_cases_path / test_case_path_stem
    expected_model_yaml = test_case_path / "expected.yaml"
    # Setup the pyPESTO model selector instance.
    petab_select_problem = petab_select.Problem.from_yaml(
        test_case_path / "petab_select_problem.yaml",
    )
    pypesto_select_problem = pypesto.select.Problem(
        petab_select_problem=petab_select_problem
    )

    # Run the selection process until "exhausted".
    pypesto_select_problem.select_to_completion(
        model_problem_options=model_problem_options,
    )

    # Get the best model
    best_model = petab_select.analyze.get_best(
        models=pypesto_select_problem.calibrated_models,
        criterion=petab_select_problem.criterion,
        compare=petab_select_problem.compare,
    )

    # Load the expected model.
    expected_model = Model.from_yaml(expected_model_yaml)

    def get_series(model, dict_attribute) -> pd.Series:
        return pd.Series(
            getattr(model, dict_attribute),
            dtype=np.float64,
        ).sort_index()

    # The estimated parameters and criteria values are as expected.
    for dict_attribute in [CRITERIA, ESTIMATED_PARAMETERS]:
        pd.testing.assert_series_equal(
            get_series(expected_model, dict_attribute),
            get_series(best_model, dict_attribute),
            rtol=1e-2,
        )
    # FIXME ensure `current model criterion` trajectory also matches, in summary.tsv file,
    #       for test case 0009, after summary format is revised


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Too CPU heavy for CI.",
)
def test_famos_cli():
    """Run test case 0009 with pyPESTO and the CLI interface."""
    test_case_path = test_cases_path / "0009"
    expected_model_yaml = test_case_path / "expected.yaml"
    problem_yaml = test_case_path / "petab_select_problem.yaml"

    problem = petab_select.Problem.from_yaml(problem_yaml)

    # Setup working directory for intermediate files
    work_dir = Path("output_famos_cli")
    work_dir_str = str(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir_str)
    work_dir.mkdir(exist_ok=True, parents=True)

    models_yamls = []
    metadata_yaml = work_dir / "metadata.yaml"
    state_dill = work_dir / "state.dill"
    iteration = 0
    while True:
        iteration += 1
        uncalibrated_models_yaml = (
            work_dir / f"uncalibrated_models_{iteration}.yaml"
        )
        calibrated_models_yaml = (
            work_dir / f"calibrated_models_{iteration}.yaml"
        )
        models_yaml = work_dir / f"models_{iteration}.yaml"
        models_yamls.append(models_yaml)
        # Start iteration
        subprocess.run(
            shlex.split(  # noqa: S603
                f"""petab_select start_iteration
                    --problem {problem_yaml}
                    --state {state_dill}
                    --output-uncalibrated-models {uncalibrated_models_yaml}
                    --relative-paths
                """
            )
        )
        # Calibrate models
        models = petab_select.Models.from_yaml(uncalibrated_models_yaml)
        for model in models:
            pypesto.select.ModelProblem(
                model=model,
                criterion=problem.criterion,
                **model_problem_options,
            )
        models.to_yaml(filename=calibrated_models_yaml)
        # End iteration
        subprocess.run(
            shlex.split(  # noqa: S603
                f"""petab_select end_iteration
                    --output-models {models_yaml}
                    --output-metadata {metadata_yaml}
                    --state {state_dill}
                    --calibrated-models {calibrated_models_yaml}
                    --relative-paths
                """
            )
        )
        with open(metadata_yaml) as f:
            metadata = yaml.safe_load(f)
        if metadata[TERMINATE]:
            break

    # Get the best model
    models_yamls_arg = " ".join(
        f"--models {models_yaml}" for models_yaml in models_yamls
    )
    subprocess.run(
        shlex.split(  # noqa: S603
            f"""petab_select get_best
                --problem {problem_yaml}
                {models_yamls_arg}
                --output {work_dir / "best_model.yaml"}
                --relative-paths
            """
        )
    )
    best_model = petab_select.Model.from_yaml(work_dir / "best_model.yaml")

    # Load the expected model.
    expected_model = Model.from_yaml(expected_model_yaml)

    def get_series(model, dict_attribute) -> pd.Series:
        return pd.Series(
            getattr(model, dict_attribute),
            dtype=np.float64,
        ).sort_index()

    # The estimated parameters and criteria values are as expected.
    for dict_attribute in [CRITERIA, ESTIMATED_PARAMETERS]:
        pd.testing.assert_series_equal(
            get_series(expected_model, dict_attribute),
            get_series(best_model, dict_attribute),
            rtol=1e-2,
        )
    # FIXME ensure `current model criterion` trajectory also matches, in summary.tsv file,
    # after summary format is revised
