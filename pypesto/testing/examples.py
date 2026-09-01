"""Example problems for tests / demos."""

import pandas as pd

from pypesto.C import InnerParameterType


def get_Boehm_JProteomeRes2014_hierarchical_petab() -> (
    "petab.Problem"  # noqa: F821
):
    """
    Get Boehm_JProteomeRes2014 problem with scaled/offset observables.

    Creates a modified version of the Boehm_JProteomeRes2014 benchmark problem,
    suitable for hierarchical optimization.
    """
    import petab.v1 as petab
    from benchmark_models_petab import get_problem

    from pypesto.C import PARAMETER_TYPE

    petab_problem = get_problem("Boehm_JProteomeRes2014")
    # Add scaling and offset parameters
    petab_problem.observable_df[petab.OBSERVABLE_FORMULA] = [
        f"observableParameter2_{obs_id} + observableParameter1_{obs_id} "
        f"* {obs_formula}"
        for obs_id, obs_formula in zip(
            petab_problem.observable_df.index,
            petab_problem.observable_df[petab.OBSERVABLE_FORMULA],
            strict=True,
        )
    ]
    # Set scaling and offset parameters for measurements
    if not (
        petab_problem.measurement_df[petab.OBSERVABLE_PARAMETERS].isna().all()
    ):
        raise AssertionError()
    petab_problem.measurement_df[petab.OBSERVABLE_PARAMETERS] = [
        f"scaling_{obs_id};offset_{obs_id}"
        for obs_id in petab_problem.measurement_df[petab.OBSERVABLE_ID]
    ]
    # Add output parameters to parameter table
    extra_parameters = [
        {
            petab.PARAMETER_ID: par_id,
            petab.PARAMETER_SCALE: petab.LIN,
            petab.LOWER_BOUND: -100,
            petab.UPPER_BOUND: 100,
            petab.NOMINAL_VALUE: 0,
            petab.ESTIMATE: 1,
        }
        for par_id in (
            "offset_pSTAT5A_rel",
            "offset_pSTAT5B_rel",
            "offset_rSTAT5A_rel",
        )
    ]

    extra_parameters.extend(
        {
            petab.PARAMETER_ID: par_id,
            petab.PARAMETER_SCALE: petab.LIN,
            petab.LOWER_BOUND: 1e-5,
            petab.UPPER_BOUND: 1e5,
            petab.NOMINAL_VALUE: nominal_value,
            petab.ESTIMATE: 1,
        }
        for par_id, nominal_value in zip(
            (
                "scaling_pSTAT5A_rel",
                "scaling_pSTAT5B_rel",
                "scaling_rSTAT5A_rel",
            ),
            (3.85261197844677, 6.59147818673419, 3.15271275648527),
            strict=True,
        )
    )

    petab_problem.parameter_df = pd.concat(
        [
            petab_problem.parameter_df,
            petab.get_parameter_df(pd.DataFrame(extra_parameters)),
        ]
    )
    # Mark output parameters for hierarchical optimization
    petab_problem.parameter_df[PARAMETER_TYPE] = None
    for par_id in petab_problem.parameter_df.index:
        if par_id.startswith("offset_"):
            petab_problem.parameter_df.loc[par_id, PARAMETER_TYPE] = (
                InnerParameterType.OFFSET.value
            )
        elif par_id.startswith("sd_"):
            petab_problem.parameter_df.loc[par_id, PARAMETER_TYPE] = (
                InnerParameterType.SIGMA.value
            )
        elif par_id.startswith("scaling_"):
            petab_problem.parameter_df.loc[par_id, PARAMETER_TYPE] = (
                InnerParameterType.SCALING.value
            )

    # log-scaling is not supported for inner parameters
    petab_problem.parameter_df.loc[
        petab_problem.parameter_df.index.str.startswith("sd_"),
        petab.PARAMETER_SCALE,
    ] = petab.LIN

    petab.lint_problem(petab_problem)

    return petab_problem


def get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds() -> (
    "petab.Problem"  # noqa: F821
):
    """
    See `get_Boehm_JProteomeRes2014_hierarchical_petab`.

    Adjusts this PEtab problem to have correct bounds for inner parameters.
    """
    from ..hierarchical.petab import correct_parameter_df_bounds

    petab_problem = get_Boehm_JProteomeRes2014_hierarchical_petab()
    petab_problem.parameter_df = correct_parameter_df_bounds(
        parameter_df=petab_problem.parameter_df
    )
    return petab_problem


def get_Boehm_JProteomeRes2014_hierarchical_petab_v2() -> (
    "petab.v2.Problem"  # noqa: F821
):
    """
    Get the PEtab v2 version of the hierarchical Boehm_JProteomeRes2014 problem.

    The PEtab v1 problem from
    `get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds` upgraded
    to PEtab v2. The pyPESTO-specific hierarchical optimization annotations
    (`parameterType` column) are preserved as extra fields of the PEtab v2
    parameter table.
    """
    import tempfile
    from pathlib import Path

    from petab import v2

    petab_problem_v1 = (
        get_Boehm_JProteomeRes2014_hierarchical_petab_corrected_bounds()
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        petab_problem_v1.to_files_generic(prefix_path=temp_dir)
        # `from_yaml` upgrades PEtab v1 problems to v2 on the fly;
        #  the returned problem is fully loaded into memory
        return v2.Problem.from_yaml(Path(temp_dir) / "problem.yaml")
