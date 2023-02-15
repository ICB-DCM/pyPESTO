"""Definition of an optimal scaling parameter class."""
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...C import (
    INNER_PARAMETER_BOUNDS,
    LIN,
    MEASUREMENT_CATEGORY,
    MEASUREMENT_GROUP,
    MEASUREMENT_TYPE,
    ORDINAL,
    REDUCED,
    STANDARD,
    TIME,
    InnerParameterType,
)
from ..problem import (
    InnerProblem,
    _get_timepoints_with_replicates,
    ix_matrices_from_arrays,
)
from .parameter import OptimalScalingParameter

try:
    import amici
    import petab
    from petab.C import OBSERVABLE_ID
except ImportError:
    pass


class OptimalScalingProblem(InnerProblem):
    """Inner optimization problem for optimal scaling.

    Attributes
    ----------
    xs:
        Mapping of (inner) parameter ID to ``InnerParameters``.
    data:
        Measurement data. One matrix (`num_timepoints` x `num_observables`)
        per simulation condition. Missing observations as NaN.
    groups:
        A dictionary of the groups of the subproblem.
    method:
        A string representing the method of the Optimal Scaling approach, either 'reduced' or 'standard'.
    """

    def __init__(
        self,
        xs: List[OptimalScalingParameter],
        data: List[np.ndarray],
        method: str,
    ):
        """Construct."""
        super().__init__(xs, data)
        self.groups = {}
        self.method = method

        self._initialize_groups()

    def _initialize_groups(self) -> None:
        """Initialize the groups of the subproblem."""
        self.groups = {}

        for group in self.get_groups_for_xs(
            InnerParameterType.OPTIMAL_SCALING
        ):
            self.groups[group] = {}
            xs = self.get_xs_for_group(group)
            self.groups[group]['num_categories'] = len(
                {x.category for x in xs}
            )
            self.groups[group]['num_datapoints'] = sum(
                [
                    sum([np.sum(ixs) for ixs in x.ixs])
                    for x in self.get_cat_ub_parameters_for_group(group)
                ]
            )

            self.groups[group]['surrogate_data'] = np.zeros(
                self.groups[group]['num_datapoints']
            )

            self.groups[group]['num_inner_params'] = (
                self.groups[group]['num_datapoints']
                + 2 * self.groups[group]['num_categories']
            )

            self.groups[group]['num_constr_full'] = (
                2 * self.groups[group]['num_datapoints']
                + 2 * self.groups[group]['num_categories']
            )

            self.groups[group]['lb_indices'] = list(
                range(
                    self.groups[group]['num_datapoints'],
                    self.groups[group]['num_datapoints']
                    + self.groups[group]['num_categories'],
                )
            )

            self.groups[group]['ub_indices'] = list(
                range(
                    self.groups[group]['num_datapoints']
                    + self.groups[group]['num_categories'],
                    self.groups[group]['num_inner_params'],
                )
            )

            self.groups[group]['C'] = self.initialize_c(group)

            self.groups[group]['W'] = self.initialize_w(group)

            self.groups[group]['Wdot'] = self.initialize_w(group)

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
        method: str = None,
    ) -> 'OptimalScalingProblem':
        """Construct the inner problem from the `petab_problem`."""
        if not method:
            method = REDUCED

        return optimal_scaling_inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas, method
        )

    def get_groups_for_xs(self, inner_parameter_type: str) -> List[int]:
        """Get unique list of ``OptimalScalingParameter.group`` values."""
        groups = [x.group for x in self.get_xs_for_type(inner_parameter_type)]
        return list(set(groups))

    def get_xs_for_group(self, group: int) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that belong to the given group."""
        return [x for x in self.xs.values() if x.group == group]

    def get_free_xs_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are free and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is True
        ]

    def get_fixed_xs_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are fixed and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.estimate is False
        ]

    def get_cat_ub_parameters_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are category upper boundaries and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == 'cat_ub'
        ]

    def get_cat_lb_parameters_for_group(
        self, group: int
    ) -> List[OptimalScalingParameter]:
        """Get ``OptimalScalingParameter``s that are category lower boundaries and belong to the given group."""
        return [
            x
            for x in self.xs.values()
            if x.group == group and x.inner_parameter_id[:6] == 'cat_lb'
        ]

    def initialize_c(self, group: int) -> np.ndarray:
        """Initialize the constraints matrix for the group.

        The structure of the constraints matrix is the following: Each row C_i of the matrix C
        represents one optimization constraint as C_i * xi + d(theta, sim) <= 0, where xi is the
        vector of inner paramters (surrogate data, lower bounds, upper bounds)^T, and d is the
        vector of minimal category interval ranges and gaps.

        First `self.groups[group]['num_datapoints']` rows constrain the surrogate data to stay
        larger than lower category bounds.
        Then another `self.groups[group]['num_datapoints']` rows constrain the surrogate data to
        stay smaller than upper category bounds.
        Then `self.groups[group]['num_categories']` rows constrain the ordering of the categories.
        And lastly, the remaining `self.groups[group]['num_categories']` constrain the lower
        bound to be smaller than the upper bound for each category.
        """
        constr = np.zeros(
            [
                self.groups[group]['num_constr_full'],
                self.groups[group]['num_inner_params'],
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
                    data_idx, cat_idx + self.groups[group]['num_datapoints']
                ] = 1

                # y_surr - ub <= 0
                constr[
                    data_idx + self.groups[group]['num_datapoints'], data_idx
                ] = 1
                constr[
                    data_idx + self.groups[group]['num_datapoints'],
                    cat_idx
                    + self.groups[group]['num_datapoints']
                    + self.groups[group]['num_categories'],
                ] = -1
                data_idx += 1

            # Constrain the ordering wrt. neighbouring categories, i.e. ub_i - lb_{i+1} <= 0.
            if cat_idx == 0:
                constr[
                    2 * self.groups[group]['num_datapoints'] + cat_idx,
                    self.groups[group]['num_datapoints'] + cat_idx,
                ] = -1
            else:
                constr[
                    2 * self.groups[group]['num_datapoints'] + cat_idx,
                    self.groups[group]['num_datapoints'] + cat_idx,
                ] = -1
                constr[
                    2 * self.groups[group]['num_datapoints'] + cat_idx,
                    self.groups[group]['num_datapoints']
                    + self.groups[group]['num_categories']
                    + cat_idx
                    - 1,
                ] = 1

            # Constrain upper bound to be larger than lower bound, i.e. lb_i - ub_i <= 0.
            constr[
                2 * self.groups[group]['num_datapoints']
                + self.groups[group]['num_categories']  # - 1
                + cat_idx,
                self.groups[group]['lb_indices'][cat_idx],
            ] = 1
            constr[
                2 * self.groups[group]['num_datapoints']
                + self.groups[group]['num_categories']  # - 1
                + cat_idx,
                self.groups[group]['ub_indices'][cat_idx],
            ] = -1

        return constr

    def initialize_w(self, group: int) -> np.ndarray:
        """Initialize the weight matrix for the group."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[group]['num_datapoints']),
                    np.zeros(2 * self.groups[group]['num_categories']),
                ]
            )
        )
        return weights

    def get_w(self, group: int, y_sim_all: np.ndarray) -> np.ndarray:
        """Return the weight matrix of the group."""
        weights = np.diag(
            np.block(
                [
                    np.ones(self.groups[group]['num_datapoints'])
                    / (np.sum(np.abs(y_sim_all)) + 1e-8),
                    np.zeros(2 * self.groups[group]['num_categories']),
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
                    np.ones(self.groups[group]['num_datapoints'])
                    * (
                        -1
                        * np.sum(sy_all)
                        / ((np.sum(np.abs(y_sim_all)) + 1e-8) ** 2)
                    ),
                    np.zeros(2 * self.groups[group]['num_categories']),
                ]
            )
        )
        return w_dot

    def get_d(
        self,
        group,
        xs: List[OptimalScalingParameter],
        y_sim_all: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """Return vector of minimal gaps and ranges."""
        max_simulation = np.nanmax(y_sim_all)

        interval_range = max_simulation / (2 * len(xs) + 1)
        interval_gap = max_simulation / (4 * (len(xs) - 1) + 1)

        d = np.zeros(self.groups[group]['num_constr_full'])

        d[
            2 * self.groups[group]['num_datapoints']
            + 1 : 2 * self.groups[group]['num_datapoints']
            + self.groups[group]['num_categories']
        ] = (interval_gap + eps)

        d[
            2 * self.groups[group]['num_datapoints']
            + self.groups[group]['num_categories'] :
        ] = interval_range
        return d

    def get_dd_dtheta(
        self,
        group: int,
        xs: List[OptimalScalingParameter],
        y_sim_all: np.ndarray,
        sy_all: np.ndarray,
    ) -> np.ndarray:
        """Return the derivative of vector of minimal gaps and ranges with respect to an outer parameter."""
        max_sim_idx = np.argmax(y_sim_all)
        max_sy = sy_all[max_sim_idx]
        dd_dtheta = np.zeros(self.groups[group]['num_constr_full'])

        dinterval_range_dtheta = max_sy / (2 * len(xs) + 1)
        dinterval_gap_dtheta = max_sy / (4 * (len(xs) - 1) + 1)

        dd_dtheta[
            2 * self.groups[group]['num_datapoints']
            + 1 : 2 * self.groups[group]['num_datapoints']
            + self.groups[group]['num_categories']
        ] = dinterval_gap_dtheta

        dd_dtheta[
            2 * self.groups[group]['num_datapoints']
            + self.groups[group]['num_categories'] :
        ] = dinterval_range_dtheta

        return dd_dtheta

    def get_inner_parameter_dictionary(self) -> Dict:
        """Return a dictionary with inner parameter ids and their values."""
        inner_par_dict = {}
        for x_id, x in self.xs.items():
            inner_par_dict[x_id] = x.value
        return inner_par_dict


def optimal_scaling_inner_problem_from_petab_problem(
    petab_problem: petab.Problem,
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
    method: str,
):
    """Construct the inner problem from the `petab_problem`."""
    # inner parameters
    inner_parameters = optimal_scaling_inner_parameters_from_measurement_df(
        petab_problem.measurement_df, method
    )

    # used indices for all measurement specific parameters
    ixs = optimal_scaling_ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, inner_parameters
    )

    # transform experimental data
    edatas = [
        amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
    ]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, edatas)

    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]

    return OptimalScalingProblem(
        inner_parameters,
        edatas,
        method,
    )


def optimal_scaling_inner_parameters_from_measurement_df(
    df: pd.DataFrame,
    method: str,
) -> List[OptimalScalingParameter]:
    """Create list of inner free parameters from PEtab measurement table dependent on the method provided."""
    df = df.reset_index()

    estimate = get_estimate_for_method(method)
    par_types = ['cat_lb', 'cat_ub']

    inner_parameters = []
    lb, ub = INNER_PARAMETER_BOUNDS[
        InnerParameterType.OPTIMAL_SCALING
    ].values()

    for par_type, par_estimate in zip(par_types, estimate):
        for _, row in df.iterrows():
            if row[MEASUREMENT_TYPE] == ORDINAL:
                par_id = f'{par_type}_{row[OBSERVABLE_ID]}_{row[MEASUREMENT_GROUP]}_{row[MEASUREMENT_CATEGORY]}'

                # Create only one set of bound parameters per category of a
                # group.
                if par_id not in [
                    inner_par.inner_parameter_id
                    for inner_par in inner_parameters
                ]:
                    inner_parameters.append(
                        OptimalScalingParameter(
                            inner_parameter_id=par_id,
                            inner_parameter_type=InnerParameterType.OPTIMAL_SCALING,
                            scale=LIN,
                            lb=lb,
                            ub=ub,
                            category=row[MEASUREMENT_CATEGORY],
                            group=row[MEASUREMENT_GROUP],
                            estimate=par_estimate,
                        )
                    )
    inner_parameters.sort(key=lambda x: (x.group, x.category))

    return inner_parameters


def get_estimate_for_method(method: str) -> Tuple[bool, bool]:
    """Return which inner parameters to estimate dependent on the method provided."""
    estimate_ub = True
    estimate_lb = False

    if method == STANDARD:
        estimate_lb = True

    return estimate_lb, estimate_ub


def optimal_scaling_ixs_for_measurement_specific_parameters(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    inner_parameters: List[OptimalScalingParameter],
) -> Dict[str, List[Tuple[int, int, int]]]:
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
    for condition_ix, condition in simulation_conditions.iterrows():
        # measurement table for current condition
        df_for_condition = petab.get_rows_for_condition(
            measurement_df=petab_problem.measurement_df, condition=condition
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
                    measurement, inner_parameters
                )

                # try to insert if hierarchical parameter
                for override in inner_par_ids_for_meas:
                    ixs_for_par.setdefault(override, []).append(
                        (condition_ix, time_w_reps_ix, observable_ix)
                    )
    return ixs_for_par


def get_inner_par_ids_for_measurement(
    measurement: Dict,
    inner_parameters: List[OptimalScalingParameter],
):
    """Return inner parameter ids of parameters which are related to the measurement."""
    return [
        inner_par.inner_parameter_id
        for inner_par in inner_parameters
        if inner_par.category == measurement[MEASUREMENT_CATEGORY]
        and inner_par.group == measurement[MEASUREMENT_GROUP]
    ]
