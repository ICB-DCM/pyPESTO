"""Definition of an optimal scaling parameter class."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...C import (
    C_MATRIX,
    CAT_LB,
    CAT_UB,
    CENSORED,
    CENSORING_BOUNDS,
    CENSORING_TYPES,
    INNER_PARAMETER_BOUNDS,
    INTERVAL_CENSORED,
    LB_INDICES,
    LEFT_CENSORED,
    LIN,
    MEASUREMENT_CATEGORY,
    MEASUREMENT_TYPE,
    NUM_CATEGORIES,
    NUM_CONSTR_FULL,
    NUM_DATAPOINTS,
    NUM_INNER_PARAMS,
    ORDINAL,
    QUANTITATIVE_DATA,
    QUANTITATIVE_IXS,
    REDUCED,
    RIGHT_CENSORED,
    STANDARD,
    SURROGATE_DATA,
    TIME,
    UB_INDICES,
    W_DOT_MATRIX,
    W_MATRIX,
    InnerParameterType,
)
from ..base_problem import (
    AmiciInnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .parameter import OrdinalParameter

try:
    import amici
    import petab.v1 as petab
    from petab.v1.C import OBSERVABLE_ID, PARAMETER_SEPARATOR
except ImportError:
    pass


class OrdinalProblem(AmiciInnerProblem):
    r"""Inner optimization problem for ordinal or censored data.

    The ordinal inner problem contains the following parameters: surrogate data,
    lower bounds, and upper bounds. All parameters are optimized to minimize the
    distance between the surrogate data and the simulated data while satisfying
    the ordering constraints of the problem. Depending on the method, the problem
    is re-formulated to reduce the number of parameters to estimate.

    Attributes
    ----------
    xs:
        Mapping of (inner) parameter ID to ``InnerParameters``.
    data:
        Measurement data. One matrix (``num_timepoints`` x ``num_observables``)
        per simulation condition. Missing observations as NaN.
    edatas:
        AMICI ``ExpData``\s for each simulation condition.
    groups:
        A dictionary of the groups of the subproblem.
    method:
        A string representing the method of the Optimal Scaling approach, either ``reduced`` or ``standard``.
    """

    def __init__(
        self,
        method: str,
        **kwargs,
    ):
        """Construct."""
        super().__init__(**kwargs)
        self.groups = {}
        self.method = method

        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize the groups of the subproblem."""
        self.groups = {}

        for group in self.get_groups_for_xs(InnerParameterType.ORDINAL):
            self.groups[group] = {}
            xs = self.get_xs_for_group(group)
            self.groups[group][NUM_CATEGORIES] = len({x.category for x in xs})
            self.groups[group][NUM_DATAPOINTS] = sum(
                [
                    sum([np.sum(ixs) for ixs in x.ixs])
                    for x in self.get_cat_ub_parameters_for_group(group)
                ]
            )

            self.groups[group][SURROGATE_DATA] = np.zeros(
                self.groups[group][NUM_DATAPOINTS]
            )

            self.groups[group][NUM_INNER_PARAMS] = (
                self.groups[group][NUM_DATAPOINTS]
                + 2 * self.groups[group][NUM_CATEGORIES]
            )

            self.groups[group][LB_INDICES] = list(
                range(
                    self.groups[group][NUM_DATAPOINTS],
                    self.groups[group][NUM_DATAPOINTS]
                    + self.groups[group][NUM_CATEGORIES],
                )
            )

            self.groups[group][UB_INDICES] = list(
                range(
                    self.groups[group][NUM_DATAPOINTS]
                    + self.groups[group][NUM_CATEGORIES],
                    self.groups[group][NUM_INNER_PARAMS],
                )
            )

            if all(x.censoring_type is not None for x in xs):
                self.groups[group][MEASUREMENT_TYPE] = CENSORED
                self.groups[group][QUANTITATIVE_IXS] = (
                    self.get_censored_group_quantitative_ixs(xs)
                )
                self.groups[group][QUANTITATIVE_DATA] = np.concatenate(
                    [
                        data_i[mask_i]
                        for data_i, mask_i in zip(
                            self.data, self.groups[group][QUANTITATIVE_IXS]
                        )
                    ]
                )
            elif all(x.censoring_type is None for x in xs):
                self.groups[group][MEASUREMENT_TYPE] = ORDINAL

                self.groups[group][NUM_CONSTR_FULL] = (
                    2 * self.groups[group][NUM_DATAPOINTS]
                    + 2 * self.groups[group][NUM_CATEGORIES]
                )

                self.groups[group][C_MATRIX] = self.initialize_c(group)

                self.groups[group][W_MATRIX] = self.initialize_w(group)

                self.groups[group][W_DOT_MATRIX] = self.initialize_w(group)
            else:
                raise ValueError(
                    "Censoring types of optimal scaling parameters of a group "
                    "have to either be all None, or all not None."
                )

    def initialize(self) -> None:
        """Initialize the subproblem."""
        # Initialize all parameter values.
        for x in self.xs.values():
            x.initialize()

        # Initialize the groups.
        for group in self.get_groups_for_xs(InnerParameterType.SPLINE):
            self.groups[group][SURROGATE_DATA] = np.zeros(
                self.groups[group][NUM_DATAPOINTS]
            )

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: amici.Model,
        edatas: list[amici.ExpData],
        method: str = None,
    ) -> OrdinalProblem:
        """Construct the inner problem from the `petab_problem`."""
        if not method:
            method = REDUCED

        return optimal_scaling_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas, method
        )

    def get_interpretable_x_ids(self) -> list[str]:
        """Get IDs of interpretable inner parameters.

        There are no interpretable inner parameters for the ordinal problem.
        """
        return []

    def get_interpretable_x_scales(self) -> list[str]:
        """Get scales of interpretable inner parameters.

        There are no interpretable inner parameters for the ordinal problem.
        """
        return []

    def get_groups_for_xs(self, inner_parameter_type: str) -> list[int]:
        """Get unique list of ``OptimalScalingParameter.group`` values."""
        groups = [x.group for x in self.get_xs_for_type(inner_parameter_type)]
        return list(set(groups))

    def get_xs_for_group(self, group: int) -> list[OrdinalParameter]:
        r"""Get ``OptimalScalingParameter``\s that belong to the given group."""
        return [x for x in self.xs.values() if x.group == group]

    def get_free_xs_for_group(self, group: int) -> list[OrdinalParameter]:
        r"""Get ``OptimalScalingParameter``\s that are free and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is True
        ]

    def get_fixed_xs_for_group(self, group: int) -> list[OrdinalParameter]:
        r"""Get ``OptimalScalingParameter``\s that are fixed and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is False
        ]

    def get_cat_ub_parameters_for_group(
        self, group: int
    ) -> list[OrdinalParameter]:
        r"""Get ``OptimalScalingParameter``\s that are category upper boundaries and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == CAT_UB
        ]

    def get_cat_lb_parameters_for_group(
        self, group: int
    ) -> list[OrdinalParameter]:
        r"""Get ``OptimalScalingParameter``\s that are category lower boundaries and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == CAT_LB
        ]

    def initialize_c(self, group: int) -> np.ndarray:
        """Initialize the constraints matrix for the group.

        The structure of the constraints matrix is the following: Each row C_i of the matrix C
        represents one optimization constraint as C_i * xi + d(theta, sim) <= 0, where xi is the
        vector of inner paramters (surrogate data, lower bounds, upper bounds)^T, and d is the
        vector of minimal category interval ranges and gaps.

        First `self.groups[group][NUM_DATAPOINTS]` rows constrain the surrogate data to stay
        larger than lower category bounds.
        Then another `self.groups[group][NUM_DATAPOINTS]` rows constrain the surrogate data to
        stay smaller than upper category bounds.
        Then `self.groups[group][NUM_CATEGORIES]` rows constrain the ordering of the categories.
        And lastly, the remaining `self.groups[group][NUM_CATEGORIES]` constrain the lower
        bound to be smaller than the upper bound for each category.
        """
        constr = np.zeros(
            [
                self.groups[group][NUM_CONSTR_FULL],
                self.groups[group][NUM_INNER_PARAMS],
            ]
        )
        data_idx = 0

        # Iterate over categories.
        for cat_idx, category in enumerate(
            self.get_cat_ub_parameters_for_group(group)
        ):
            num_data_in_cat = int(
                np.sum(
                    [
                        np.sum(category.ixs[idx])
                        for idx in range(len(category.ixs))
                    ]
                )
            )

            # Constrain the surrogate data of this category to stay within it.
            for _ in range(num_data_in_cat):
                # lb - y_surr <= 0
                constr[data_idx, data_idx] = -1
                constr[
                    data_idx, cat_idx + self.groups[group][NUM_DATAPOINTS]
                ] = 1

                # y_surr - ub <= 0
                constr[
                    data_idx + self.groups[group][NUM_DATAPOINTS], data_idx
                ] = 1
                constr[
                    data_idx + self.groups[group][NUM_DATAPOINTS],
                    cat_idx
                    + self.groups[group][NUM_DATAPOINTS]
                    + self.groups[group][NUM_CATEGORIES],
                ] = -1
                data_idx += 1

            # Constrain the ordering wrt. neighbouring categories, i.e. ub_i - lb_{i+1} <= 0.
            if cat_idx == 0:
                constr[
                    2 * self.groups[group][NUM_DATAPOINTS] + cat_idx,
                    self.groups[group][NUM_DATAPOINTS] + cat_idx,
                ] = -1
            else:
                constr[
                    2 * self.groups[group][NUM_DATAPOINTS] + cat_idx,
                    self.groups[group][NUM_DATAPOINTS] + cat_idx,
                ] = -1
                constr[
                    2 * self.groups[group][NUM_DATAPOINTS] + cat_idx,
                    self.groups[group][NUM_DATAPOINTS]
                    + self.groups[group][NUM_CATEGORIES]
                    + cat_idx
                    - 1,
                ] = 1

            # Constrain upper bound to be larger than lower bound, i.e. lb_i - ub_i <= 0.
            constr[
                2 * self.groups[group][NUM_DATAPOINTS]
                + self.groups[group][NUM_CATEGORIES]  # - 1
                + cat_idx,
                self.groups[group][LB_INDICES][cat_idx],
            ] = 1
            constr[
                2 * self.groups[group][NUM_DATAPOINTS]
                + self.groups[group][NUM_CATEGORIES]  # - 1
                + cat_idx,
                self.groups[group][UB_INDICES][cat_idx],
            ] = -1

        return constr

    def initialize_w(self, group: int) -> np.ndarray:
        """Initialize the weight matrix for the group."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[group][NUM_DATAPOINTS]),
                    np.zeros(2 * self.groups[group][NUM_CATEGORIES]),
                ]
            )
        )
        return weights

    def get_w(self, group: int, y_sim_all: np.ndarray) -> np.ndarray:
        """Return the weight matrix of the group."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[group][NUM_DATAPOINTS])
                    / (np.sum(np.abs(y_sim_all)) + 1e-8),
                    np.zeros(2 * self.groups[group][NUM_CATEGORIES]),
                ]
            )
        )
        return weights

    def get_wdot(
        self, group: int, y_sim_all: np.ndarray, sy_all: np.ndarray
    ) -> np.ndarray:
        """Return the derivative of the weight matrix of a group with respect to an outer parameter."""
        w_dot = np.diag(
            np.block(
                [
                    np.ones(self.groups[group][NUM_DATAPOINTS])
                    * (
                        -1
                        * np.sum(sy_all)
                        / ((np.sum(np.abs(y_sim_all)) + 1e-8) ** 2)
                    ),
                    np.zeros(2 * self.groups[group][NUM_CATEGORIES]),
                ]
            )
        )
        return w_dot

    def get_d(
        self,
        group,
        xs: list[OrdinalParameter],
        y_sim_all: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Return vector of minimal gaps and ranges."""
        max_simulation = np.nanmax(y_sim_all)

        interval_range = max_simulation / (2 * len(xs) + 1)
        interval_gap = max_simulation / (4 * (len(xs) - 1) + 1)

        d = np.zeros(self.groups[group][NUM_CONSTR_FULL])

        d[
            2 * self.groups[group][NUM_DATAPOINTS] + 1 : 2
            * self.groups[group][NUM_DATAPOINTS]
            + self.groups[group][NUM_CATEGORIES]
        ] = interval_gap + eps

        d[
            2 * self.groups[group][NUM_DATAPOINTS]
            + self.groups[group][NUM_CATEGORIES] :
        ] = interval_range
        return d

    def get_dd_dtheta(
        self,
        group: int,
        xs: list[OrdinalParameter],
        y_sim_all: np.ndarray,
        sy_all: np.ndarray,
    ) -> np.ndarray:
        """Return the derivative of vector of minimal gaps and ranges with respect to an outer parameter."""
        max_sim_idx = np.argmax(y_sim_all)
        max_sy = sy_all[max_sim_idx]
        dd_dtheta = np.zeros(self.groups[group][NUM_CONSTR_FULL])

        dinterval_range_dtheta = max_sy / (2 * len(xs) + 1)
        dinterval_gap_dtheta = max_sy / (4 * (len(xs) - 1) + 1)

        dd_dtheta[
            2 * self.groups[group][NUM_DATAPOINTS] + 1 : 2
            * self.groups[group][NUM_DATAPOINTS]
            + self.groups[group][NUM_CATEGORIES]
        ] = dinterval_gap_dtheta

        dd_dtheta[
            2 * self.groups[group][NUM_DATAPOINTS]
            + self.groups[group][NUM_CATEGORIES] :
        ] = dinterval_range_dtheta

        return dd_dtheta

    def get_censored_group_quantitative_ixs(
        self, xs: list[OrdinalParameter]
    ) -> list[np.ndarray]:
        r"""Return a list of boolean masks indicating which data points are quantitative.

        For a given group with censored data, return a list of boolean masks indicating
        which data points are not censored, and therefore quantitative.

        Parameters
        ----------
        xs:
            List of ``OptimalScalingParameter``\s of a group with censored data.

        Returns
        -------
        quantitative_ixs:
            List of boolean masks indicating which data points are quantitative.
        """
        # Initialize boolean masks with False and find corresponding observable index.
        quantitative_ixs = [np.full(ixs_i.shape, False) for ixs_i in xs[0].ixs]
        observable_index = xs[0].group - 1

        # Set to True all datapoints of the corresponding observable.
        if np.ndim(quantitative_ixs) == 2:
            quantitative_ixs = [
                np.full(ixs_i.shape, True) for ixs_i in xs[0].ixs
            ]
        else:
            for quantitative_ixs_i in quantitative_ixs:
                quantitative_ixs_i[:, observable_index] = True

        # Set to False for all censored datapoints.
        for x in xs:
            for ixs_i, quantitative_ixs_i in zip(x.ixs, quantitative_ixs):
                quantitative_ixs_i[ixs_i] = False

        return quantitative_ixs

    def get_inner_parameter_dictionary(self) -> dict:
        """Return a dictionary with inner parameter ids and their values."""
        inner_par_dict = {}
        for x_id, x in self.xs.items():
            inner_par_dict[x_id] = x.value
        return inner_par_dict


def optimal_scaling_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: amici.Model,
    edatas: list[amici.ExpData],
    method: str,
):
    """Construct the inner problem from the `petab_problem`."""
    # inner parameters
    inner_parameters = optimal_scaling_inner_parameters_from_measurement_df(
        petab_problem.measurement_df, method, amici_model
    )

    # used indices for all measurement specific parameters
    ixs = optimal_scaling_ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, inner_parameters
    )

    # transform experimental data
    data = [amici.numpy.ExpDataView(edata)["observedData"] for edata in edatas]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]

    return OrdinalProblem(
        xs=inner_parameters,
        data=data,
        edatas=edatas,
        method=method,
    )


def optimal_scaling_inner_parameters_from_measurement_df(
    df: pd.DataFrame,
    method: str,
    amici_model: amici.Model,
) -> list[OrdinalParameter]:
    """Create list of inner free parameters from PEtab measurement table dependent on the method provided."""
    df = df.reset_index()

    observable_ids = amici_model.getObservableIds()

    estimate = get_estimate_for_method(method)
    par_types = [CAT_LB, CAT_UB]

    inner_parameters = []
    lb, ub = INNER_PARAMETER_BOUNDS[InnerParameterType.ORDINAL].values()

    for observable_idx, observable_id in enumerate(observable_ids):
        group = observable_idx + 1

        observable_df = df[df[OBSERVABLE_ID] == observable_id]

        if all(observable_df[MEASUREMENT_TYPE] == ORDINAL):
            # Add optimal scaling parameters for ordinal measurements.
            for par_type, par_estimate in zip(par_types, estimate):
                for _, row in observable_df.iterrows():
                    par_id = f"{par_type}_{observable_id}_{row[MEASUREMENT_TYPE]}_{int(row[MEASUREMENT_CATEGORY])}"

                    # Create only one set of bound parameters per category of a group.
                    if par_id not in [
                        inner_par.inner_parameter_id
                        for inner_par in inner_parameters
                    ]:
                        inner_parameters.append(
                            OrdinalParameter(
                                inner_parameter_id=par_id,
                                inner_parameter_type=InnerParameterType.ORDINAL,
                                scale=LIN,
                                lb=lb,
                                ub=ub,
                                observable_id=observable_id,
                                category=int(row[MEASUREMENT_CATEGORY]),
                                group=group,
                                estimate=par_estimate,
                            )
                        )
        elif any(observable_df[MEASUREMENT_TYPE].isin(CENSORING_TYPES)):
            # Get df with only censored measurements.
            censored_df = observable_df.loc[
                observable_df[MEASUREMENT_TYPE].isin(CENSORING_TYPES)
            ]
            # Check for unique values in the CENSORING_BOUNDS column and
            # order them with resect to the first float value in the string.
            unique_censoring_bounds = sorted(
                censored_df[CENSORING_BOUNDS].unique(),
                key=lambda x: float(str(x).split(PARAMETER_SEPARATOR)[0]),
            )
            for par_type in par_types:
                for _, row in censored_df.iterrows():
                    category = int(
                        unique_censoring_bounds.index(row[CENSORING_BOUNDS])
                        + 1
                    )
                    par_id = f"{par_type}_{observable_id}_{row[MEASUREMENT_TYPE]}_{category}"
                    # Create only one set of bound parameters per category of a group.
                    if par_id not in [
                        inner_par.inner_parameter_id
                        for inner_par in inner_parameters
                    ]:
                        inner_parameters.append(
                            OrdinalParameter(
                                inner_parameter_id=par_id,
                                inner_parameter_type=InnerParameterType.ORDINAL,
                                scale=LIN,
                                lb=lb,
                                ub=ub,
                                observable_id=observable_id,
                                category=category,
                                group=group,
                                estimate=False,
                                censoring_type=row[MEASUREMENT_TYPE],
                            )
                        )
                        _add_value_to_censored_bound_parameter(
                            inner_parameters[-1], row, par_type
                        )

    inner_parameters.sort(key=lambda x: (x.group, x.category))

    return inner_parameters


def get_estimate_for_method(method: str) -> tuple[bool, bool]:
    """Return which inner parameters to estimate dependent on the method provided."""
    estimate_ub = True
    estimate_lb = False

    if method == STANDARD:
        estimate_lb = True

    return estimate_lb, estimate_ub


def optimal_scaling_ixs_for_measurement_specific_parameters(
    petab_problem: petab.Problem,
    amici_model: amici.Model,
    inner_parameters: list[OrdinalParameter],
) -> dict[str, list[tuple[int, int, int]]]:
    """Create mapping of parameters to measurements.

    Returns
    -------
    A dictionary mapping parameter ID to a list of
    `(condition index, time index, observable index)` tuples in which this
    output parameter is used. For each condition, the time index refers to
    a sorted list of non-unique time points for which there are measurements.
    """
    ixs_for_par = {}
    observable_ids = amici_model.getObservableIds()

    simulation_conditions = (
        petab_problem.get_simulation_conditions_from_measurement_df()
    )
    measurement_df = petab_problem.measurement_df
    # Get unique censoring bounds for each observable.
    unique_censoring_bounds_per_observable = {}
    if CENSORING_BOUNDS in measurement_df.columns:
        for observable_id in observable_ids:
            observable_df = measurement_df[
                measurement_df[OBSERVABLE_ID] == observable_id
            ]
            censored_observable_df = observable_df.loc[
                observable_df[MEASUREMENT_TYPE].isin(CENSORING_TYPES)
            ]
            unique_censoring_bounds_per_observable[observable_id] = sorted(
                censored_observable_df[CENSORING_BOUNDS].unique(),
                key=lambda x: float(str(x).split(PARAMETER_SEPARATOR)[0]),
            )

    for condition_ix, condition in simulation_conditions.iterrows():
        # measurement table for current condition
        df_for_condition = petab.get_rows_for_condition(
            measurement_df=measurement_df, condition=condition
        )

        # unique sorted list of timepoints
        timepoints = sorted(df_for_condition[TIME].unique().astype(float))
        # non-unique sorted list of timepoints
        timepoints_w_reps = _get_timepoints_with_replicates(
            measurement_df=df_for_condition
        )

        for time in timepoints:
            # subselect measurements for time `time`
            df_for_time = df_for_condition[df_for_condition[TIME] == time]
            time_ix_0 = timepoints_w_reps.index(time)

            # remember used time indices for each observable
            time_ix_for_obs_ix = {}

            # iterate over measurements
            for _, measurement in df_for_time.iterrows():
                # extract observable index
                observable_ix = observable_ids.index(
                    measurement[OBSERVABLE_ID]
                )

                # as the time indices have to account for replicates, we need
                #  to track which time indices have already been assigned for
                #  the current observable
                if observable_ix in time_ix_for_obs_ix:
                    # a replicate
                    time_ix_for_obs_ix[observable_ix] += 1
                else:
                    # the first measurement for this `(observable, timepoint)`
                    time_ix_for_obs_ix[observable_ix] = time_ix_0
                time_w_reps_ix = time_ix_for_obs_ix[observable_ix]

                inner_par_ids_for_meas = get_inner_par_ids_for_measurement(
                    measurement,
                    inner_parameters,
                    unique_censoring_bounds_per_observable,
                )

                # try to insert if hierarchical parameter
                if inner_par_ids_for_meas:
                    for override in inner_par_ids_for_meas:
                        ixs_for_par.setdefault(override, []).append(
                            (condition_ix, time_w_reps_ix, observable_ix)
                        )
    return ixs_for_par


def get_inner_par_ids_for_measurement(
    measurement: dict,
    inner_parameters: list[OrdinalParameter],
    unique_censoring_bounds_per_observable: dict[str, list[float]],
):
    """Return inner parameter ids of parameters which are related to the measurement."""
    if measurement[MEASUREMENT_TYPE] == ORDINAL:
        return [
            inner_par.inner_parameter_id
            for inner_par in inner_parameters
            if inner_par.category == measurement[MEASUREMENT_CATEGORY]
            and inner_par.observable_id == measurement[OBSERVABLE_ID]
        ]
    elif measurement[MEASUREMENT_TYPE] in [
        LEFT_CENSORED,
        INTERVAL_CENSORED,
        RIGHT_CENSORED,
    ]:
        unique_censoring_bounds = unique_censoring_bounds_per_observable[
            measurement[OBSERVABLE_ID]
        ]
        measurement_category = int(
            unique_censoring_bounds.index(measurement[CENSORING_BOUNDS]) + 1
        )
        return [
            inner_par.inner_parameter_id
            for inner_par in inner_parameters
            if inner_par.observable_id == measurement[OBSERVABLE_ID]
            and inner_par.category == measurement_category
        ]


def _add_value_to_censored_bound_parameter(
    inner_parameter: OrdinalParameter,
    row: pd.Series,
    par_type: str,
) -> None:
    if row[MEASUREMENT_TYPE] == LEFT_CENSORED and par_type == CAT_LB:
        inner_parameter.value = 0.0
    elif row[MEASUREMENT_TYPE] == LEFT_CENSORED and par_type == CAT_UB:
        inner_parameter.value = float(row[CENSORING_BOUNDS])

    elif row[MEASUREMENT_TYPE] == RIGHT_CENSORED and par_type == CAT_LB:
        inner_parameter.value = float(row[CENSORING_BOUNDS])
    elif row[MEASUREMENT_TYPE] == RIGHT_CENSORED and par_type == CAT_UB:
        inner_parameter.value = np.inf

    elif row[MEASUREMENT_TYPE] == INTERVAL_CENSORED and par_type == CAT_LB:
        inner_parameter.value = float(
            row[CENSORING_BOUNDS].split(PARAMETER_SEPARATOR)[0]
        )
    elif row[MEASUREMENT_TYPE] == INTERVAL_CENSORED and par_type == CAT_UB:
        inner_parameter.value = float(
            row[CENSORING_BOUNDS].split(PARAMETER_SEPARATOR)[1]
        )
