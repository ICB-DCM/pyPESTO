from __future__ import annotations

import numbers
from typing import Sequence

import numpy as np
import pandas as pd
import petab
from petab.C import LIN, LOG, LOG10


class ExpData:
    """Class for managing experimental data for a single condition."""

    def __init__(
        self,
        condition_id: str,
        measurements: np.ndarray,
        timepoints: Sequence[float],
        observable_ids: Sequence[str],
        noise_distributions: np.ndarray,
        noise_formulae: np.ndarray,
        measurement_df: pd.DataFrame,
    ):
        """
        Initialize the ExpData object.

        Parameters
        ----------
        condition_id:
            Identifier of the condition.
        measurements:
            Numpy Array containing the measurement data. It is a 2D array of
            dimension (n_timepoints, n_observables + 1). The first column is
            the timepoints, the remaining columns are the observable values.
            Observables not measured at a given timepoint should be NaN.
        timepoints:
            Timepoints of the measurement data.
        observable_ids:
            Observable ids of the measurement data. Order should match the
            columns of the measurements array (-time).
        noise_distributions:
            Numpy Array describing noise distributions of the measurement
            data. Dimension: (n_timepoints, n_observables). Each entry is a
            string describing scale and type of noise distribution, the name
            is "scale_type". E.g. "lin_normal", "log_normal", "log10_normal".
        noise_formulae:
            Numpy Array describing noise formulae of the measurement data.
            Dimension: (n_timepoints, n_observables). Each entry is a string
            describing the noise formula, either a parameter name or a constant.
        """
        self.condition_id = condition_id
        self.measurements = measurements
        self.timepoints = timepoints
        self.observable_ids = observable_ids
        self.noise_distributions = noise_distributions
        self.noise_formulae = noise_formulae
        self.measurement_df = measurement_df

        # run sanity checks
        if not self.sanity_check():
            raise ValueError("Data is not sane.")

    def get_timepoints(self):
        """
        Get the timepoints of the measurement data.

        Returns
        -------
        timepoints:
            Timepoints of the measurement data.
        """
        return self.timepoints

    def get_observable_ids(self):
        """
        Get the observable ids of the measurement data.

        Returns
        -------
        observable_ids:
            Observable ids of the measurement data.
        """
        return self.observable_ids

    def sanity_check(self):
        """
        Perform a sanity check of the data.

        Returns
        -------
        sanity_check:
            True if the data is sane, False otherwise.
        """
        # TODO: needs to be implemented
        return True

    @staticmethod
    def from_petab_problem(petab_problem: petab.Problem) -> Sequence[ExpData]:
        """
        Create a list of ExpData object from a petab problem.

        Parameters
        ----------
        petab_problem:
            PEtab problem.
        """
        # extract all condition ids from measurement data
        condition_ids = list(
            set(petab_problem.measurement_df["simulationConditionId"].values)
        )
        exp_datas = [
            ExpData.from_petab_single_condition(
                condition_id=condition_id, petab_problem=petab_problem
            )
            for condition_id in condition_ids
        ]
        return exp_datas

    @staticmethod
    def from_petab_single_condition(
        condition_id: str, petab_problem: petab.Problem
    ) -> ExpData:
        """
        Create an ExpData object from a single condition of a petab problem.

        Parameters
        ----------
        condition_id:
            Identifier of the condition.
        measurement_df:
            DataFrame containing the measurement data of only
        petab_problem:
            PEtab problem.
        """
        # extract measurement data for a single condition
        measurement_df = petab_problem.measurement_df[
            petab_problem.measurement_df["simulationConditionId"]
            == condition_id
        ]
        # turn measurement data into a numpy array
        measurements, observale_ids = measurement_df_to_matrix(measurement_df)
        # timepoints are the first column of the measurements array
        timepoints = measurements[:, 0]
        # TODO: perhaps needs to be revised at a later stage
        timepoints = sorted(set(map(float, timepoints)))
        # construct noise distributions and noise formulae
        noise_distributions, noise_formulae = construct_noise_matrices(
            petab_problem, observale_ids, condition_id
        )
        return ExpData(
            condition_id=condition_id,
            measurements=measurements,
            timepoints=timepoints,
            observable_ids=observale_ids,
            noise_distributions=noise_distributions,
            noise_formulae=noise_formulae,
            measurement_df=measurement_df,
        )


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


def measurement_df_to_matrix(
    measurement_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Convert a PEtab measurement DataFrame to a matrix.

    Parameters
    ----------
    measurement_df:
        DataFrame containing the measurement data.

    Returns
    -------
    measurement_matrix:
        Numpy array containing the measurement data. It is a 2D array of
        dimension (n_timepoints, n_observables + 1). The first column is
        the timepoints, the remaining columns are the observable values.
        Observables not measured at a given timepoint should be NaN.
    observable_ids:
        Observable ids of the measurement data.
    """
    # select only 'observableId', 'time', and 'measurement' columns
    measurement_df = measurement_df[["observableId", "time", "measurement"]]
    # add a count to get replicates
    measurement_df["count"] = measurement_df.groupby(
        ["observableId", "time"]
    ).cumcount()
    # pivot the DataFrame to have unique observables as columns, index=time
    pivot_df = measurement_df.pivot(
        index=["time", "count"],
        columns="observableId",
        values="measurement",
    ).fillna(np.nan)
    # reset the index
    pivot_df.reset_index(inplace=True)
    # drop the count column
    pivot_df.drop(columns="count", inplace=True)
    # get observable ids
    observable_ids = pivot_df.columns[1:]
    # convert the pivoted DataFrame to a NumPy array
    measurement_matrix = pivot_df.to_numpy()

    return measurement_matrix, list(observable_ids)


def construct_noise_matrices(
    petab_problem: petab.Problem,
    observable_ids: Sequence[str],
    condition_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct noise matrices from a PEtab problem.

    Parameters
    ----------
    petab_problem:
        PEtab problem.
    observable_ids:
        Observable ids of the measurement data.
    condition_id:
        Identifier of the condition.

    Returns
    -------
    noise_distributions:
        Numpy Array describing noise distributions of the measurement
        data. Dimension: (n_timepoints, n_observables). Each entry is a
        string describing scale and type of noise distribution, the name
        is "scale_type". E.g. "lin_normal", "log_normal", "log10_normal".
    noise_formulae:
        Numpy Array describing noise formulae of the measurement data.
        Dimension: (n_timepoints, n_observables). Each entry is a string
        describing the noise formula, either a parameter name or a constant.
    """

    def _get_noise(
        petab_problem: petab.Problem, observable_id: str, condition_id: str
    ) -> tuple[str, str]:
        """
        Get noise distribution and noise formula for a single observable.

        Parameters
        ----------
        petab_problem:
            PEtab problem.
        observable_id:
            Identifier of the observable.
        condition_id:
            Identifier of the condition.

        Returns
        -------
        noise:
            Tuple of noise distribution and noise formula.
        """
        obs_df = petab_problem.observable_df
        # check whether Index name is "observableId", if yes, get the row
        if obs_df.index.name == "observableId":
            row = obs_df.loc[observable_id]
        elif "observableId" in obs_df.columns:
            row = obs_df[obs_df["observableId"] == observable_id].iloc[0]
        else:
            raise ValueError("No observableId in observable_df.")
        # noise distribution
        noise_scale = "lin"
        noise_type = "normal"
        # check if "observableTransformation" and "noiseDistribution" exist
        if "observableTransformation" in obs_df.columns:
            if not pd.isna(row["observableTransformation"]):
                noise_scale = row["observableTransformation"]
        if "noiseDistribution" in obs_df.columns:
            if not pd.isna(row["noiseDistribution"]):
                noise_type = row["noiseDistribution"]
        noise_distribution = f"{noise_scale}_{noise_type}"
        # TODO: check if noise_distribution is a allowed
        # noise formula
        noise_formula = row["noiseFormula"]
        return noise_distribution, noise_formula

    # extract noise distributions and noise formulae
    noise = [
        _get_noise(petab_problem, observable_id, condition_id)
        for observable_id in observable_ids
    ]

    noise_distributions, noise_formulae = zip(*noise)
    return np.array(noise_distributions), np.array(noise_formulae)
