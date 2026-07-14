"""Utility functions for working with roadrunner and PEtab.

Includes ExpData class for managing experimental data, SolverOptions class for
managing roadrunner solver options, and utility functions to convert between
PEtab measurement data and a roarunner simulation output back and forth.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

try:
    import petab.v1 as petab
    from petab.v1.C import (
        LIN,
        MEASUREMENT,
        NOISE_DISTRIBUTION,
        NOISE_FORMULA,
        NOISE_PARAMETERS,
        NORMAL,
        OBSERVABLE_ID,
        OBSERVABLE_TRANSFORMATION,
        SIMULATION,
        SIMULATION_CONDITION_ID,
        TIME,
    )
except ImportError:
    petab = None

try:
    import roadrunner
except ImportError:
    roadrunner = None


class ExpData:
    """Class for managing experimental data for a single condition."""

    def __init__(
        self,
        condition_id: str,
        measurements: np.ndarray,
        observable_ids: Sequence[str],
        noise_distributions: np.ndarray,
        noise_formulae: np.ndarray,
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
            Observable ids of the measurement data. Order must match the
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
        self.observable_ids = observable_ids
        self.noise_distributions = noise_distributions
        self.noise_formulae = noise_formulae

        self.sanity_check()

    # define timepoints as a property
    @property
    def timepoints(self):
        """Timepoints of the measurement data."""
        return self.measurements[:, 0]

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
        """Perform a sanity check of the data."""
        if self.measurements.shape[1] != len(self.observable_ids) + 1:
            raise ValueError(
                "Number of columns in measurements does not match number of "
                "observable ids + time."
            )
        # check that the noise distributions and noise formulae have the
        # correct shape: either (n_observables,) or (n_timepoints, n_observables)
        n_timepoints = self.measurements.shape[0]
        n_observables = len(self.observable_ids)

        # noise_distributions can be 1D or 2D
        if self.noise_distributions.ndim == 1:
            if len(self.noise_distributions) != n_observables:
                raise ValueError(
                    f"Number of noise distributions ({len(self.noise_distributions)}) "
                    f"does not match number of observable ids ({n_observables})."
                )
        elif self.noise_distributions.ndim == 2:
            if self.noise_distributions.shape != (n_timepoints, n_observables):
                raise ValueError(
                    f"Shape of noise distributions {self.noise_distributions.shape} "
                    f"does not match (n_timepoints, n_observables) = ({n_timepoints}, {n_observables})."
                )
        else:
            raise ValueError(
                f"noise_distributions must be 1D or 2D, got {self.noise_distributions.ndim}D"
            )

    @staticmethod
    def from_petab_problem(petab_problem: petab.Problem) -> list[ExpData]:
        """
        Create a list of ExpData object from a petab problem.

        Parameters
        ----------
        petab_problem:
            PEtab problem.
        """
        # extract all condition ids from measurement data
        condition_ids = list(
            petab_problem.measurement_df["simulationConditionId"].unique()
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
        petab_problem:
            PEtab problem.
        """
        # extract measurement data for a single condition
        measurement_df = petab_problem.measurement_df[
            petab_problem.measurement_df[SIMULATION_CONDITION_ID]
            == condition_id
        ]
        # turn measurement data into a numpy array
        measurements, observale_ids = measurement_df_to_matrix(measurement_df)
        # construct noise distributions and noise formulae
        noise_distributions, noise_formulae = construct_noise_matrices(
            petab_problem, observale_ids
        )
        # inject timepoint-specific noise parameter overrides if present
        noise_distributions, noise_formulae = inject_timepoint_specific_noise(
            noise_distributions,
            noise_formulae,
            measurement_df,
            observale_ids,
            measurements,
        )
        return ExpData(
            condition_id=condition_id,
            measurements=measurements,
            observable_ids=observale_ids,
            noise_distributions=noise_distributions,
            noise_formulae=noise_formulae,
        )


class SolverOptions(dict):
    """Class for managing solver options of roadrunner."""

    def __init__(
        self,
        integrator: str | None = None,
        relative_tolerance: float | None = None,
        absolute_tolerance: float | None = None,
        maximum_num_steps: int | None = None,
        **kwargs,
    ):
        """
        Initialize the SolverOptions object. Can be used as a dictionary.

        Parameters
        ----------
        integrator:
            Integrator to use.
        relative_tolerance:
            Relative tolerance of the integrator.
        absolute_tolerance:
            Absolute tolerance of the integrator.
        maximum_num_steps:
            Maximum number of steps to take.
        kwargs:
            Additional solver options.
        """
        super().__init__()
        if integrator is None:
            integrator = "cvode"
        self.integrator = integrator
        if relative_tolerance is None:
            relative_tolerance = 1e-6
        self.relative_tolerance = relative_tolerance
        if absolute_tolerance is None:
            absolute_tolerance = 1e-12
        self.absolute_tolerance = absolute_tolerance
        if maximum_num_steps is None:
            maximum_num_steps = 20000
        self.maximum_num_steps = maximum_num_steps
        self.update(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        """Return a dict representation of the SolverOptions object."""
        return f"{self.__class__.__name__}({super().__repr__()})"

    def apply_to_roadrunner(self, roadrunner_instance: roadrunner.RoadRunner):
        """
        Apply the solver options to a roadrunner object inplace.

        Parameters
        ----------
        roadrunner_instance:
            Roadrunner object to apply the solver options to.
        """
        # don't allow 'gillespie' integrator
        if self.integrator == "gillespie":
            raise ValueError("Gillespie integrator is not supported.")
        # copy the options
        options = self.copy()
        # set integrator and remove integrator from options
        roadrunner_instance.setIntegrator(options.pop("integrator"))
        integrator = roadrunner_instance.getIntegrator()
        # set the remaining options
        for key, value in options.items():
            # try to set the options, if it fails, raise a warning
            try:
                integrator.setValue(key, value)
            except RuntimeError as e:
                warnings.warn(
                    f"Failed to set option {key} to {value}. Reason: {e}. "
                    f"Valid keys are: {integrator.getSettings()}.",
                    stacklevel=2,
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
        value_dict[key] = petab.parameters.unscale(
            value, petab_scale_dict[key]
        )

    return value_dict


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
        Observables not measured at a given timepoint will be NaN.
    observable_ids:
        Observable ids of the measurement data.
    """
    measurement_df = measurement_df.loc[
        :, ["observableId", "time", "measurement"]
    ]
    # get potential replicates via placeholder "count"
    measurement_df["count"] = measurement_df.groupby(
        ["observableId", "time"]
    ).cumcount()
    pivot_df = measurement_df.pivot(
        index=["time", "count"],
        columns="observableId",
        values="measurement",
    ).fillna(np.nan)
    pivot_df.reset_index(inplace=True)
    pivot_df.drop(columns="count", inplace=True)

    observable_ids = pivot_df.columns[1:]
    measurement_matrix = pivot_df.to_numpy()

    return measurement_matrix, list(observable_ids)


def construct_noise_matrices(
    petab_problem: petab.Problem,
    observable_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct noise matrices from a PEtab problem.

    Parameters
    ----------
    petab_problem:
        PEtab problem.
    observable_ids:
        Observable ids of the measurement data.

    Returns
    -------
    noise_distributions:
        Numpy Array describing noise distributions of the measurement
        data. Dimension: (1, n_observables). Each entry is a
        string describing scale and type of noise distribution, the name
        is "scale_type". E.g. "lin_normal", "log_normal", "log10_normal".
    noise_formulae:
        Numpy Array describing noise formulae of the measurement data.
        Dimension: (1, n_observables). Each entry is a string
        describing the noise formula, either a parameter name or a constant.
    """

    def _get_noise(observable_id: str) -> tuple[str, str]:
        """
        Get noise distribution and noise formula for a single observable.

        Parameters
        ----------
        observable_id:
            Identifier of the observable.

        Returns
        -------
        noise:
            Tuple of noise distribution and noise formula.
        """
        obs_df = petab_problem.observable_df
        # check whether Index name is "observableId", if yes, get the row
        if obs_df.index.name == OBSERVABLE_ID:
            row = obs_df.loc[observable_id]
        elif OBSERVABLE_ID in obs_df.columns:
            row = obs_df[obs_df[OBSERVABLE_ID] == observable_id].iloc[0]
        else:
            raise ValueError("No observableId in observable_df.")
        # noise distribution
        noise_scale = LIN
        noise_type = NORMAL
        # check if "observableTransformation" and "noiseDistribution" exist
        if OBSERVABLE_TRANSFORMATION in obs_df.columns:
            if not pd.isna(row[OBSERVABLE_TRANSFORMATION]):
                noise_scale = row[OBSERVABLE_TRANSFORMATION]
        if NOISE_DISTRIBUTION in obs_df.columns:
            if not pd.isna(row[NOISE_DISTRIBUTION]):
                noise_type = row[NOISE_DISTRIBUTION]
        noise_distribution = f"{noise_scale}_{noise_type}"
        # TODO: check if noise_distribution is a allowed
        # noise formula
        noise_formula = row[NOISE_FORMULA]
        return noise_distribution, noise_formula

    # extract noise distributions and noise formulae
    noise = [_get_noise(observable_id) for observable_id in observable_ids]

    noise_distributions, noise_formulae = zip(*noise, strict=True)
    return np.array(noise_distributions), np.array(noise_formulae)


def inject_timepoint_specific_noise(
    noise_distributions: np.ndarray,
    noise_formulae: np.ndarray,
    measurement_df: pd.DataFrame,
    observable_ids: Sequence[str],
    measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inject timepoint-specific noise parameter overrides from measurement table.

    Takes the default noise distributions and formulae (one per observable) and
    expands them to timepoint-specific arrays if the measurement table contains
    timepoint-specific noise parameter overrides.

    Parameters
    ----------
    noise_distributions:
        Default noise distributions from observable table. Shape: (n_observables,)
    noise_formulae:
        Default noise formulae from observable table. Shape: (n_observables,)
    measurement_df:
        PEtab measurement DataFrame filtered for a single condition.
    observable_ids:
        Observable IDs in the order they appear in the measurement matrix.
    measurements:
        Measurement matrix with shape (n_timepoints, n_observables + 1).
        First column is time.

    Returns
    -------
    noise_distributions:
        Noise distributions array. Shape: (n_timepoints, n_observables)
    noise_formulae:
        Noise formulae array with timepoint-specific overrides applied.
        Shape: (n_timepoints, n_observables)
    """
    # Get timepoints from measurements
    timepoints = measurements[:, 0]
    n_timepoints = len(timepoints)

    # Initialize arrays with default values repeated for each timepoint
    noise_distributions_expanded = np.tile(
        noise_distributions, (n_timepoints, 1)
    )
    noise_formulae_expanded = np.tile(noise_formulae, (n_timepoints, 1))

    # If no noise parameters column, return the 1D defaults
    if NOISE_PARAMETERS not in measurement_df.columns:
        return noise_distributions, noise_formulae

    # Check if there are actually timepoint-specific numeric overrides
    # Build a lookup dictionary: (time, observable_id) -> noise_parameter_value
    noise_override_map = {}
    has_timepoint_specific = False

    for _, row in measurement_df.iterrows():
        time = row[TIME]
        obs_id = row[OBSERVABLE_ID]
        noise_param = row[NOISE_PARAMETERS]
        # Only process non-NaN values
        if pd.notna(noise_param):
            noise_param_str = str(noise_param)
            # Check if it's a simple numeric value (not a formula like "0.5;2")
            try:
                float(noise_param_str)
                # It's numeric - add to map
                noise_override_map[(time, obs_id)] = noise_param_str
            except ValueError:
                # Not a simple number (could be "0.5;2", "noise", etc.)
                # Don't treat as timepoint-specific override
                pass

    # Check if noise values actually vary by timepoint for any observable
    if noise_override_map:
        for obs_id in observable_ids:
            obs_noise_values = [
                v for (t, o), v in noise_override_map.items() if o == obs_id
            ]
            if len(set(obs_noise_values)) > 1:
                # This observable has different noise values at different timepoints
                has_timepoint_specific = True
                break

    # If no timepoint-specific overrides, return 1D defaults
    if not has_timepoint_specific:
        return noise_distributions, noise_formulae

    # Apply timepoint-specific overrides
    for i_time, time in enumerate(timepoints):
        for i_obs, obs_id in enumerate(observable_ids):
            key = (time, obs_id)
            if key in noise_override_map:
                # Override the noise formula with the value from measurement table
                noise_formulae_expanded[i_time, i_obs] = noise_override_map[
                    key
                ]

    return noise_distributions_expanded, noise_formulae_expanded


def simulation_to_measurement_df(
    simulations: dict,
    measurement_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert simulation results to a measurement DataFrame.

    Parameters
    ----------
    simulations:
        Dictionary containing the simulation results of a roadrunner
        simulator. The keys are the condition ids and the values are the
        simulation results.
    measurement_df:
        DataFrame containing the measurement data of the PEtab problem.
    """
    simulation_conditions = petab.get_simulation_conditions(measurement_df)
    meas_dfs = []
    for _, condition_id in simulation_conditions.iterrows():
        meas_df_cond = measurement_df[
            measurement_df[SIMULATION_CONDITION_ID]
            == condition_id[SIMULATION_CONDITION_ID]
        ]
        sim_res = simulations[condition_id[SIMULATION_CONDITION_ID]]
        # in each row, replace the "measurement" with the simulation value
        for index, row in meas_df_cond.iterrows():
            timepoint = row[TIME]
            observable_id = row[OBSERVABLE_ID]
            time_index = np.where(sim_res[TIME] == timepoint)[0][0]
            sim_value = sim_res[observable_id][time_index]
            meas_df_cond.at[index, MEASUREMENT] = sim_value
        # rename measurement to simulation
        meas_df_cond = meas_df_cond.rename(columns={MEASUREMENT: SIMULATION})
        meas_dfs.append(meas_df_cond)
    sim_res_df = pd.concat(meas_dfs)
    return sim_res_df
