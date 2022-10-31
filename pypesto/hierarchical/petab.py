"""Helper methods for hierarchical optimization with PEtab."""

import copy

import pandas as pd
import petab
from petab.C import OBSERVABLE_ID

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

    # Loop over each observable
    for (
        observable_id,
        observable_def,
    ) in petab_problem.observable_df.iterrows():
        inner_parameters_observable = {
            inner_parameter_type: set()
            for inner_parameter_type in InnerParameterType
        }

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
