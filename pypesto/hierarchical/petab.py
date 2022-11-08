"""Helper methods for hierarchical optimization with PEtab."""

import copy

import pandas as pd
import petab
import sympy as sp
from more_itertools import one
from petab.C import (
    LIN,
    NOISE_DISTRIBUTION,
    NOISE_FORMULA,
    NOISE_PARAMETERS,
    NORMAL,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_TRANSFORMATION,
    PARAMETER_SEPARATOR,
)

from ..C import PARAMETER_TYPE, InnerParameterType


def validate_petab(petab_problem: petab.Problem) -> None:
    """Validate a PEtab problem for hierarchical optimization.

    Parameters
    ----------
    petab_problem:
        The PEtab problem.
    """
    assigned_ids = set()

    # TODO could optionally only collect estimated inner parameters,
    #      may break workflows that modify inner parameters online.
    inner_parameter_types = {
        inner_parameter_id: inner_parameter_type
        for inner_parameter_type in InnerParameterType
        for inner_parameter_id in petab_problem.parameter_df.loc[
            petab_problem.parameter_df[PARAMETER_TYPE] == inner_parameter_type
        ].index
    }

    # Check measurements to ensure sigma parameters do not appear there.
    # Ideally scaling and offset parameters would not either.
    # FIXME: In principle, sigma parameters can appear in the measurements
    # table, but makes the validation slightly more effortful.
    noise_parameters = [
        noise_parameter
        for noise_parameter_list in petab_problem.measurement_df[
            NOISE_PARAMETERS
        ]
        for noise_parameter in noise_parameter_list.split(PARAMETER_SEPARATOR)
    ]
    for noise_parameter in noise_parameters:
        # Loops over all inner parameters. Could limit this to just sigma.
        for inner_parameter_id in inner_parameter_types:
            if inner_parameter_id == noise_parameter:
                raise ValueError(
                    f"The inner parameter `{inner_parameter_id}` exists in "
                    "`noiseParameters` column of the measurements table. This "
                    "may be valid, but is not checked by this validator. "
                    "Please reformulate the problem such that this parameter "
                    "instead appears in the `noiseFormula` column of the "
                    "observables table."
                )

    # Loop over each observable
    for (
        observable_id,
        observable_def,
    ) in petab_problem.observable_df.iterrows():
        inner_parameters_observable = {
            inner_parameter_type: set()
            for inner_parameter_type in InnerParameterType
        }

        # Check that objective function has no observable transformation,
        # uses a normally-distributed noise model, and the sigma parameters
        # are the "full" noise formula.
        noise_formula = observable_def.get(NOISE_FORMULA, 1)
        for (
            inner_parameter_id,
            inner_parameter_type,
        ) in inner_parameter_types.items():
            if inner_parameter_type != InnerParameterType.SIGMA:
                continue
            if (
                inner_parameter_id in noise_formula
                and noise_formula != inner_parameter_id
            ):
                # The case where the sigma parameter is subbed into the noise
                # formula from the measurements table via `noiseParameter...`
                # is handled later in this function, where such sigma
                # parameters are banned from appearing in the measurements
                # table `noiseParameters` column.
                raise ValueError(
                    "Hierarchically-optimized sigma parameters must be "
                    "the only term in the noise formula. This is not the case "
                    f"for observable `{observable_id}`, with noise formula "
                    f"`{noise_formula}`."
                )
        if (
            noise_distribution := observable_def.get(
                NOISE_DISTRIBUTION, NORMAL
            )
            != NORMAL
        ):
            raise ValueError(
                "Hierarchical optimization currently only works with "
                "normally-distributed noise. The observable "
                f"`{observable_id}` has the incompatible noise distribution "
                f"`{noise_distribution}`."
            )
        if (
            observable_transformation := observable_def.get(
                OBSERVABLE_TRANSFORMATION, LIN
            )
            != LIN
        ):
            raise ValueError(
                "Hierarchical optimization currently only works with "
                "untransformed objective functions. The observable "
                f"`{observable_id}` has the incompatible transformation "
                f"`{observable_transformation}`."
            )

        # Check that observable formulae are of the correct form
        observable_formula = observable_def.get(OBSERVABLE_FORMULA)
        observable_formula_sp = sp.sympify(observable_formula)
        observable_id_sp = sp.sympify(observable_id)
        formula_symbols = observable_formula_sp.free_symbols
        formula_inner_parameters = [
            str(s) for s in formula_symbols if str(s) in inner_parameter_types
        ]
        if formula_inner_parameters:
            formula_inner_scalings = [
                s
                for s in formula_inner_parameters
                if inner_parameter_types[s] == InnerParameterType.SCALING
            ]
            formula_inner_offsets = [
                s
                for s in formula_inner_parameters
                if inner_parameter_types[s] == InnerParameterType.OFFSET
            ]
            formula_inner_sigmas = [
                s
                for s in formula_inner_parameters
                if inner_parameter_types[s] == InnerParameterType.SIGMA
            ]
            if formula_inner_sigmas:
                raise ValueError(
                    "An observable formula contains sigma parameter(s). "
                    f"Sigma parameters(s): `{formula_inner_sigmas}`. "
                    f"Observable formula: `{observable_formula}`"
                )
            if (
                len(formula_inner_scalings) > 1
                or len(formula_inner_offsets) > 1
            ):
                raise ValueError(
                    "An observable formula contains multiple scaling or "
                    "offset parameters. "
                    f"Scalings: `{formula_inner_scalings}`. "
                    f"Offsets: `{formula_inner_offsets}`. "
                    f"Observable formula: `{observable_formula}`."
                )
            # Scaling and offset
            if formula_inner_scalings and formula_inner_offsets:
                scaling_id = sp.sympify(one(formula_inner_scalings))
                offset_id = sp.sympify(one(formula_inner_offsets))
            # Only scaling
            elif formula_inner_scalings:
                scaling_id = sp.sympify(one(formula_inner_scalings))
                offset_id = sp.core.singleton.S.Zero
            elif formula_inner_offsets:
                scaling_id = sp.core.singleton.S.One
                offset_id = sp.sympify(one(formula_inner_offsets))
            else:
                raise ValueError(
                    "Unknown error while parsing observable formula "
                    f"`{observable_formula}` with inner parameters "
                    f"`{formula_inner_parameters}`."
                )

            expected_formula_sp = scaling_id * observable_id_sp + offset_id
            if sp.simplify(observable_formula_sp - expected_formula_sp) != 0:
                raise ValueError(
                    "The `observableFormula` for the observable "
                    f"`{observable_id}` isn't of the correct form for "
                    "hierarchical optimization. Expected "
                    f"formula: `{expected_formula_sp}`. Actual "
                    f"formula: `{observable_formula_sp}`."
                )

        # Collect inner parameters from observable and noise formulae
        observable_row_df = pd.DataFrame(observable_def).transpose()
        output_parameter_ids = petab.observables.get_output_parameters(
            observable_df=observable_row_df,
            model=petab_problem.model,
            observables=True,
            noise=True,
        )
        for output_parameter_id in output_parameter_ids:
            if output_parameter_id not in inner_parameter_types:
                continue
            inner_parameters_observable[
                inner_parameter_types[output_parameter_id]
            ].add(output_parameter_id)

        # Collect and check inner parameters for each measurement of the
        # observable
        assigned_observable_inner_parameters = {
            inner_parameter_type: set()
            for inner_parameter_type in InnerParameterType
        }
        for _, measurement_def in petab_problem.measurement_df.loc[
            petab_problem.measurement_df[OBSERVABLE_ID] == observable_id
        ].iterrows():
            inner_parameters_measurement = copy.deepcopy(
                inner_parameters_observable
            )

            for (
                measurement_parameter_id
            ) in petab.measurements.get_measurement_parameter_ids(
                measurement_df=pd.DataFrame(measurement_def).transpose(),
            ):
                if measurement_parameter_id not in inner_parameter_types:
                    continue
                inner_parameters_measurement[
                    inner_parameter_types[measurement_parameter_id]
                ].add(measurement_parameter_id)

            # Each measurement should only be associated with, at most, a
            # single offset, scaling and sigma parameter.
            # Each measurement of the same observable must have the same
            # estimated scaling, offset, and sigma parameters.
            for inner_parameter_type in InnerParameterType:
                assigned_observable_inner_parameters[
                    inner_parameter_type
                ] |= inner_parameters_measurement[inner_parameter_type]

        for (
            inner_parameter_type,
            inner_parameter_ids,
        ) in assigned_observable_inner_parameters.items():
            if len(inner_parameter_ids) > 1:
                raise ValueError(
                    f'The observable `{observable_id}` has measurements with '
                    'different estimated inner parameters of type '
                    f'`{inner_parameter_type}` and IDs '
                    f'`{inner_parameter_ids}`.'
                )

            if duplicate := assigned_ids.intersection(inner_parameter_ids):
                raise ValueError(
                    'The same inner parameter appears to be used across '
                    f'different observables: `{duplicate}`.'
                )

            assigned_ids.update(inner_parameter_ids)
