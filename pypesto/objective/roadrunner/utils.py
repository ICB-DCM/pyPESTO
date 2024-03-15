import numbers

import numpy as np
import pandas as pd
from petab.C import LIN, LOG, LOG10


class ExpData:
    """Class for managing experimental data for a single condition."""

    def __init__(self, condition_id: str, measurement_df: pd.DataFrame):
        """
        Initialize the ExpData object.

        Parameters
        ----------
        condition_id:
            Identifier of the condition.
        measurement_df:
            DataFrame containing the measurement data.
        """
        # TODO: add sanity checks
        self.measurement_df = measurement_df
        self.condition_id = condition_id

    def get_timepoints(self):
        """
        Get the timepoints of the measurement data.

        Returns
        -------
        timepoints:
            Timepoints of the measurement data.
        """
        return sorted(set(self.measurement_df["time"].values))

    def get_observable_ids(self):
        """
        Get the observable ids of the measurement data.

        Returns
        -------
        observable_ids:
            Observable ids of the measurement data.
        """
        return list(set(self.measurement_df["observableId"].values))


def unscale_parameters(value_dict: dict, petab_scale_dict: dict) -> dict:
    """
    Unscale the scaled parameters from target scale to linear.

    Parameters
    ----------
    value_dict:
        Dictionary with values to scale.
    petab_scale_dict:
        Target Scales.

    Returns
    -------
    unscaled_parameters:
        Dict of unscaled parameters.
    """
    if value_dict.keys() != petab_scale_dict.keys():
        raise AssertionError("Keys don't match.")

    for key, value in value_dict.items():
        value_dict[key] = unscale_parameter(value, petab_scale_dict[key])

    return value_dict


def unscale_parameter(
    value: numbers.Number, petab_scale: str
) -> numbers.Number:
    """Bring a parameter from target scale to linear scale.

    Parameters
    ----------
    value:
        Value to scale.
    petab_scale:
        Target scale of ``value``.

    Returns
    -------
    ``value`` in linear scale.
    """
    if petab_scale == LIN:
        return value
    if petab_scale == LOG10:
        return np.power(10, value)
    if petab_scale == LOG:
        return np.exp(value)
    raise ValueError(
        f"Unknown parameter scale {petab_scale}. "
        f"Must be from {(LIN, LOG, LOG10)}"
    )
