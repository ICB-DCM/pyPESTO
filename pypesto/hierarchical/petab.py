"""Helper methods for hierarchical optimization with PEtab."""

from typing import Dict, Literal, Tuple

import pandas as pd
import petab
import sympy as sp
from more_itertools import one
from petab.C import LIN
from petab.C import LOWER_BOUND as PETAB_LOWER_BOUND
from petab.C import (
    OBSERVABLE_ID,
    OBSERVABLE_TRANSFORMATION,
    PARAMETER_SEPARATOR,
)
from petab.C import UPPER_BOUND as PETAB_UPPER_BOUND
from petab.observables import get_formula_placeholders

from ..C import INNER_PARAMETER_BOUNDS
from ..C import LOWER_BOUND as PYPESTO_LOWER_BOUND
from ..C import PARAMETER_TYPE
from ..C import UPPER_BOUND as PYPESTO_UPPER_BOUND
from ..C import InnerParameterType


def correct_parameter_df_bounds(parameter_df: pd.DataFrame) -> pd.DataFrame:
    """Correct the bounds of inner parameters in a PEtab parameters table.

    Parameters
    ----------
    parameter_df:
        The PEtab parameters table.

    Returns
    -------
    The table, with corrected bounds.
    """

    def correct_row(row: pd.Series) -> pd.Series:
        if pd.isna(row[PARAMETER_TYPE]):
            return row
        bounds = INNER_PARAMETER_BOUNDS[row[PARAMETER_TYPE]]
        row[PETAB_LOWER_BOUND] = bounds[PYPESTO_LOWER_BOUND]
        row[PETAB_UPPER_BOUND] = bounds[PYPESTO_UPPER_BOUND]
        return row

    return parameter_df.apply(correct_row, axis=1)


def validate_hierarchical_petab_problem(petab_problem: petab.Problem) -> None:
    """Validate a PEtab problem for hierarchical optimization.

    Parameters
    ----------
    petab_problem:
        The PEtab problem.
    """
    if PARAMETER_TYPE not in petab_problem.parameter_df:
        # not a hierarchical optimization problem
        return

    # ensure we only have linear parameter scale
    inner_parameter_table = petab_problem.parameter_df[
        petab_problem.parameter_df[PARAMETER_TYPE].isin(
            [
                InnerParameterType.OFFSET,
                InnerParameterType.SIGMA,
                InnerParameterType.SCALING,
                InnerParameterType.OPTIMAL_SCALING,
            ]
        )
    ]
    if (
        petab.PARAMETER_SCALE in inner_parameter_table
        and not (
            inner_parameter_table[petab.PARAMETER_SCALE].isna()
            | (inner_parameter_table[petab.PARAMETER_SCALE] == petab.LIN)
        ).all()
    ):
        sub_df = inner_parameter_table.loc[
            :, [PARAMETER_TYPE, petab.PARAMETER_SCALE]
        ]
        raise NotImplementedError(
            "Only parameterScale=lin supported for parameters of the inner "
            f"subproblem.\n{sub_df}"
        )

    inner_parameter_df = validate_measurement_formulae(
        petab_problem=petab_problem
    )

    validate_inner_parameter_pairings(inner_parameter_df=inner_parameter_df)


def validate_inner_parameter_pairings(
    inner_parameter_df: pd.DataFrame,
):
    """Validate the pairings of inner parameters.

    Parameters
    ----------
    inner_parameter_df:
        Each row has the offset, scaling and sigma of each measurement in
        measurement table of the PEtab problem.
    """
    scalings_with_sigmas = {}
    offsets_with_sigmas = {}

    scalings_with_offsets = {}
    offsets_with_scalings = {}

    for _, row in inner_parameter_df.iterrows():
        offset = row[InnerParameterType.OFFSET]
        scaling = row[InnerParameterType.SCALING]
        sigma = row[InnerParameterType.SIGMA]

        # Ensures each scaling is only ever paired with one sigma.
        if scaling is not None and sigma is not None:
            expected_sigma = scalings_with_sigmas.get(scaling)
            if expected_sigma is not None and sigma != expected_sigma:
                raise ValueError(
                    "If a scaling parameter is paired with a sigma "
                    "parameter for one measurement, all measurements with "
                    "the same scaling parameter must not have a different "
                    "sigma inner parameter. "
                    f"Scaling: `{scaling}`. Sigma: `{sigma}`. "
                    f"Expected sigma: `{expected_sigma}`."
                )
            else:
                scalings_with_sigmas[scaling] = sigma

        # Ensures each offset is only ever paired with one sigma.
        if offset is not None and sigma is not None:
            expected_sigma = offsets_with_sigmas.get(offset)
            if expected_sigma is not None and sigma != expected_sigma:
                raise ValueError(
                    "If an offset parameter is paired with a sigma "
                    "parameter for one measurement, all measurements with "
                    "the same offset parameter must not have a different "
                    "sigma inner parameter. "
                    f"Scaling: `{offset}`. Sigma: `{sigma}`. "
                    f"Expected sigma: `{expected_sigma}`."
                )
            else:
                offsets_with_sigmas[offset] = sigma

        # Ensures each scaling is only ever paired with one offset, and vice
        # versa.
        if scaling is not None and offset is not None:
            expected_offset = scalings_with_offsets.get(scaling)
            expected_scaling = offsets_with_scalings.get(offset)
            if (expected_offset is not None and offset != expected_offset) or (
                expected_scaling is not None and scaling != expected_scaling
            ):
                raise ValueError(
                    "If a scaling parameter is paired with an offset "
                    "parameter for one measurement, all measurements with "
                    "the same scaling or offset parameter must not have a "
                    f"different scaling or offset inner parameter. "
                    f"Scaling: `{scaling}`. "
                    f"Expected scaling: `{expected_scaling}`. "
                    f"Offset: `{offset}`. "
                    f"Expected offset: `{expected_offset}`."
                )
            else:
                scalings_with_offsets[scaling] = offset
                offsets_with_scalings[offset] = scaling

        # TODO can a scaling that is paired with an offset, exist without
        #      any scaling?
        #      Same for offset?
        #      Same for scaling+sigma and offset+sigma pairings?


def get_inner_parameters(
    petab_problem: petab.Problem,
) -> Dict[str, InnerParameterType]:
    """Get information about the inner parameters.

    Parameters
    ----------
    petab_problem:
        The PEtab problem, with inner parameters.

    Returns
    -------
    Parameter IDs and their inner parameter types.
    """
    # TODO exclude fixed inner parameters

    if PARAMETER_TYPE not in petab_problem.parameter_df.columns:
        raise ValueError(
            "Hierarchical optimization requires specification of inner "
            f"parameters with the `{PARAMETER_TYPE}` column in the parameters "
            "table."
        )

    inner_parameters = {}
    for parameter_id, row in petab_problem.parameter_df.iterrows():
        type_str = row[PARAMETER_TYPE]
        if pd.isna(type_str):
            continue

        try:
            inner_parameters[parameter_id] = InnerParameterType(type_str)
        except ValueError as e:
            raise ValueError(
                f"Unknown inner parameter type `{type_str}`."
            ) from e

    return inner_parameters


def validate_measurement_formulae(
    petab_problem: petab.Problem,
) -> pd.DataFrame:
    """Check whether formulae associated with a measurement are valid.

    Specifically, check whether the measurement-specific observable and noise
    formulae are valid.

    Parameters
    ----------
    petab_problem:
        The PEtab problem.

    Returns
    -------
    A dataframe containing the inner parameters for each measurement.
    """
    inner_parameters = get_inner_parameters(petab_problem=petab_problem)

    inner_parameter_sets = []

    for _, measurement in petab_problem.measurement_df.iterrows():
        offset, scaling = _validate_measurement_specific_observable_formula(
            measurement=measurement,
            petab_problem=petab_problem,
            inner_parameters=inner_parameters,
        )
        sigma = _validate_measurement_specific_noise_formula(
            measurement=measurement,
            petab_problem=petab_problem,
            inner_parameters=inner_parameters,
        )
        inner_parameter_sets.append(
            [
                str(v) if v is not None else None
                for v in [offset, scaling, sigma]
            ]
        )

    return pd.DataFrame(
        data=inner_parameter_sets,
        columns=[
            InnerParameterType.OFFSET,
            InnerParameterType.SCALING,
            InnerParameterType.SIGMA,
        ],
    )


def _validate_measurement_specific_observable_formula(
    measurement: pd.Series,
    petab_problem: petab.Problem,
    inner_parameters: Dict[str, InnerParameterType],
) -> Tuple[InnerParameterType, InnerParameterType]:
    """Check whether a measurement observable formula is valid.

    Parameters
    ----------
    measurement:
        A row from a PEtab measurements table.
    petab_problem:
        The PEtab problem.
    inner_parameters:
        See `get_inner_parameters`.

    Returns
    -------
    The offset and scaling parameters.
    """
    formula, formula_inner_parameters = _get_symbolic_formula_from_measurement(
        measurement=measurement,
        formula_type='observable',
        petab_problem=petab_problem,
        inner_parameters=inner_parameters,
    )

    offset = None
    scaling = None

    terms = sp.Add.make_args(formula)

    for (
        formula_inner_parameter,
        inner_parameter_type,
    ) in formula_inner_parameters.items():
        if inner_parameter_type == InnerParameterType.OFFSET:
            offset = formula_inner_parameter
            appearances = [
                term for term in terms if offset in term.free_symbols
            ]
            try:
                # `one` ensures the offset occurs in exactly one term
                # `==`  ensures that the offset appearances as its own term,
                #       e.g. without being multiplied by anything
                if offset != one(appearances):
                    raise ValueError
            except Exception as e:
                raise ValueError(
                    "An offset is in the observable formula, but is not of "
                    "the expected form. The observable formula should take "
                    "the form `y = ... + offset`. Observable formula: "
                    f"`{formula}`. Offset: `{offset}`."
                ) from e
        elif inner_parameter_type == InnerParameterType.SCALING:
            scaling = formula_inner_parameter
            appearances = [
                term for term in terms if scaling in term.free_symbols
            ]
            # Should occur in exactly one term, and should be multiplied by
            # everything else in the term.
            try:
                # `one` ensures the scaling occurs in exactly one term
                factors = sp.Mul.make_args(one(appearances))
                factor_appearances = [
                    factor
                    for factor in factors
                    if scaling in factor.free_symbols
                ]
                # as for offset, ensure scaling is exactly one factor
                if scaling != one(factor_appearances):
                    raise ValueError
            except Exception as e:
                raise ValueError(
                    "A scaling is in the observable formula, but is not of "
                    "the expected form. The observable formula should take "
                    "the form `y = scaling*(...) [+ offset]`. Observable "
                    f"formula: `{formula}`. Offset: `{scaling}`."
                ) from e
        else:
            raise ValueError("Unknown error: unexpected inner parameter type.")

    return offset, scaling


def _validate_measurement_specific_noise_formula(
    measurement: pd.Series,
    petab_problem: petab.Problem,
    inner_parameters: Dict[str, InnerParameterType],
) -> Tuple[InnerParameterType, InnerParameterType]:
    """Check whether a measurement noise formula is valid.

    Parameters
    ----------
    measurement:
        A row from a PEtab measurements table.
    petab_problem:
        The PEtab problem.
    inner_parameters:
        See `get_inner_parameters`.

    Returns
    -------
    The sigma parameter.
    """
    formula, formula_inner_parameters = _get_symbolic_formula_from_measurement(
        measurement=measurement,
        formula_type='noise',
        petab_problem=petab_problem,
        inner_parameters=inner_parameters,
    )

    sigma = None

    for (
        formula_inner_parameter,
        inner_parameter_type,
    ) in formula_inner_parameters.items():
        if inner_parameter_type == InnerParameterType.SIGMA:
            sigma = formula_inner_parameter
            try:
                # The sigma must describe the noise completely, i.e. constitute
                # the full noise formula.
                if sigma != formula:
                    raise ValueError
            except Exception as e:
                raise ValueError(
                    "A sigma parameter does not constitute the full noise "
                    f"formula. Noise formula: `{formula}`. Sigma: `{sigma}`."
                ) from e
        else:
            raise ValueError("Unknown error: unexpected inner parameter type.")

    return sigma


def _get_symbolic_formula_from_measurement(
    measurement: pd.Series,
    formula_type: Literal['observable', 'noise'],
    petab_problem: petab.Problem,
    inner_parameters: Dict[str, InnerParameterType],
) -> Tuple[sp.Expr, Dict[sp.Symbol, InnerParameterType]]:
    """Get a symbolic representation of a formula, with overrides overridden.

    Also performs some checks to ensure only valid numbers and types of inner
    parameters are in the formula.

    Parameters
    ----------
    measurement:
        A row from the PEtab measurements table.
    formula_type:
        The formula to extract from the measurement.
    petab_problem:
        The PEtab problem.
    inner_parameters:
        See `get_inner_parameters`.

    Returns
    -------
    The symbolic formula, and its inner parameters.
    """
    observable_id = measurement[OBSERVABLE_ID]

    formula_string = petab_problem.observable_df.loc[
        observable_id, formula_type + 'Formula'
    ]
    symbolic_formula = sp.sympify(formula_string)

    formula_placeholders = get_formula_placeholders(
        formula_string=formula_string,
        observable_id=observable_id,
        override_type=formula_type,
    )
    if formula_placeholders:
        overrides = measurement[formula_type + 'Parameters']
        overrides = (
            overrides.split(PARAMETER_SEPARATOR)
            if isinstance(overrides, str)
            else [overrides]
        )
        subs = dict(zip(formula_placeholders, overrides))
        symbolic_formula = symbolic_formula.subs(subs)

    symbolic_formula_inner_parameters = {
        sp.Symbol(inner_parameter_id): inner_parameter_type
        for inner_parameter_id, inner_parameter_type in inner_parameters.items()
        if sp.Symbol(inner_parameter_id) in symbolic_formula.free_symbols
    }

    if formula_type == 'noise':
        max_parameters = 1
        expected_inner_parameter_types = [InnerParameterType.SIGMA]
    elif formula_type == 'observable':
        max_parameters = 2
        expected_inner_parameter_types = [
            InnerParameterType.OFFSET,
            InnerParameterType.SCALING,
        ]

    if len(symbolic_formula_inner_parameters) > max_parameters:
        raise ValueError(
            "There are too many inner parameters in the formula. "
            f"Formula type: `{formula_type}`. Formula: `{symbolic_formula}`. "
            f"Inner parameters: `{symbolic_formula_inner_parameters}`."
        )

    inner_parameter_types = set(symbolic_formula_inner_parameters.values())
    for inner_parameter_type in inner_parameter_types:
        if inner_parameter_type not in expected_inner_parameter_types:
            raise ValueError(
                f"Unexpected inner parameter type in {formula_type} formula. "
                f"Inner parameter type: `{inner_parameter_type}`. "
                f"Formula: `{symbolic_formula}`. Expected inner parameter "
                f"types: `{expected_inner_parameter_types}`."
            )
    if len(inner_parameter_types) != len(symbolic_formula_inner_parameters):
        raise ValueError(
            "There are multiple inner parameters of the same type."
            f"Inner parameters: `{symbolic_formula_inner_parameters.values}`."
        )

    if symbolic_formula_inner_parameters:
        observable_transformation = petab_problem.observable_df.loc[
            observable_id
        ].get(OBSERVABLE_TRANSFORMATION)
        if (
            observable_transformation is not None
            and observable_transformation != LIN
        ):
            raise ValueError(
                "Non-linear observable transformations are not supported if "
                "the observable is associated with hierarchically-optimized "
                f"inner parameters. "
                f"Observable transformation: `{observable_transformation}`. "
                f"Measurement:\n{measurement}"
            )

    return symbolic_formula, symbolic_formula_inner_parameters
