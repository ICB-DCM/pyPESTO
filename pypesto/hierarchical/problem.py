"""Inner optimization problem in hierarchical optimization."""
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .parameter import InnerParameter

try:
    import amici
    import petab
    from petab.C import (
        ESTIMATE,
        LOWER_BOUND,
        NOISE_PARAMETERS,
        OBSERVABLE_ID,
        OBSERVABLE_PARAMETERS,
        PARAMETER_ID,
        PARAMETER_SCALE,
        TIME,
        UPPER_BOUND,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)

PARAMETER_TYPE = 'parameterType'
PARAMETER_GROUP = 'parameterGroup'
PARAMETER_CATEGORY = 'parameterCategory'


class InnerProblem:
    def __init__(self, xs: List[InnerParameter], data: List[np.ndarray]):
        self.xs: Dict[str, InnerParameter] = {x.id: x for x in xs}
        self.data = data
        self._solve_numerically = False

        logger.debug(f"Created InnerProblem with ids {self.get_x_ids()}")

    @staticmethod
    def from_petab_amici(
        petab_problem: 'petab.Problem',
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
    ):
        return inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas
        )

    def get_x_ids(self) -> List[str]:
        return [x.id for x in self.xs.values()]

    def get_xs_for_type(self, type: str) -> List[InnerParameter]:
        return [x for x in self.xs.values() if x.type == type]

    def get_groups_for_xs(self, type: str) -> List[int]:
        groups = [x.group for x in self.xs.values() if x.type == type]
        return list(set(groups))

    def get_xs_for_group(self, group: int) -> List[InnerParameter]:
        return [x for x in self.xs.values() if x.group == group]

    def get_boring_pars(self, scaled: bool) -> Dict[str, float]:
        return {
            x.id: scale_value(x.boring_val, x.scale)
            if scaled
            else x.boring_val
            for x in self.xs.values()
        }

    def get_for_id(self, id: str) -> InnerParameter:
        if id in self.xs:
            return self.xs[id]
        raise KeyError(f"Cannot find id {id}.")

    def is_empty(self) -> bool:
        return len(self.xs) == 0


class AmiciInnerProblem(InnerProblem):
    def __init__(self, edatas: List[amici.ExpData], **kwargs):
        super().__init__(**kwargs)
        self.edataviews = [amici.numpy.ExpDataView(edata) for edata in edatas]

    def check_edatas(self, edatas):
        edataviews = [amici.numpy.ExpDataView(edata) for edata in edatas]

        if len(self.edataviews) != len(edataviews):
            return False

        for edataview0, edataview in zip(self.edataviews, edataviews):
            if not compare_edataviews(edataview0=edataview0, edataview=edataview):
                return False

        return True


def compare_edataviews(edataview0, edataview):
    for field_name in amici.numpy.ExpDataView._field_names:
        if edataview0[field_name] is None and edataview[field_name] is None:
            continue
        if not np.array_equal(edataview0[field_name], edataview[field_name], equal_nan=True):
            return False
    return True


def scale_value_dict(
    dct: Dict[str, float], problem: InnerProblem
) -> Dict[str, float]:
    """Scale a value dictionary."""
    scaled_dct = {}
    for key, val in dct.items():
        x = problem.get_for_id(key)
        scaled_dct[key] = scale_value(val, x.scale)
    return scaled_dct


def scale_value(
    val: Union[float, np.array], scale: str
) -> Union[float, np.array]:
    """Scale a single value."""
    if scale == 'lin':
        return val
    if scale == 'log':
        return np.log(val)
    if scale == 'log10':
        return np.log10(val)
    raise ValueError(f"Scale {scale} not recognized.")


def inner_problem_from_petab_problem(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    edatas: List['amici.ExpData'],
) -> InnerProblem:
    """
    Create inner problem from PEtab problem.

    Hierarchical optimization is a pypesto-specific PEtab extension.
    """
    import amici
    # inner parameters
    inner_parameters = inner_parameters_from_parameter_df(
        petab_problem.parameter_df
    )

    x_ids = [x.id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, x_ids
    )

    # transform experimental data
    data = [
        amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
    ]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.id]

    par_group_types = {
        tuple(obs_pars.split(';')): {
            petab_problem.parameter_df.loc[obs_par, PARAMETER_TYPE]
            for obs_par in obs_pars.split(';')
        }
        for (obs_id, obs_pars), _ in petab_problem.measurement_df.groupby(
            [petab.OBSERVABLE_ID, petab.OBSERVABLE_PARAMETERS], dropna=True
        )
        if ';' in obs_pars  # prefilter for at least 2 observable parameters
    }

    coupled_pars = {
        par
        for group, types in par_group_types.items()
        if InnerParameter.SCALING in types and InnerParameter.OFFSET
        for par in group
    }

    for par in inner_parameters:
        if par.type not in [InnerParameter.SCALING, InnerParameter.OFFSET]:
            continue
        if par.id in coupled_pars:
            par.coupled = True

    return AmiciInnerProblem(xs=inner_parameters, data=data, edatas=edatas)


def inner_parameters_from_parameter_df(
    df: pd.DataFrame,
) -> List[InnerParameter]:
    """
    Create list of inner free parameters from PEtab parameter table.

    Inner parameters are those that have a non-empty `parameterType` in the
    PEtab problem.
    """
    # create list of hierarchical parameters
    df = df.reset_index()

    for col in (PARAMETER_TYPE, PARAMETER_GROUP, PARAMETER_CATEGORY):
        if col not in df:
            df[col] = None

    parameters = []

    for _, row in df.iterrows():
        if not row[ESTIMATE]:
            continue
        if petab.is_empty(row[PARAMETER_TYPE]):
            continue
        parameters.append(
            InnerParameter(
                id=row[PARAMETER_ID],
                type=row[PARAMETER_TYPE],
                scale=row[PARAMETER_SCALE],
                lb=row[LOWER_BOUND],
                ub=row[UPPER_BOUND],
                category=row[PARAMETER_CATEGORY],
                group=row[PARAMETER_GROUP],
            )
        )

    return parameters


def ixs_for_measurement_specific_parameters(
    petab_problem: 'petab.Problem',
    amici_model: 'amici.Model',
    x_ids: List[str],
) -> Dict[str, List[Tuple[int, int, int]]]:
    """
    Create mapping of parameters to measurements.

    Returns
    -------
    A dictionary mapping parameter ID to a List of condition, time, observable
    index tuples in which this output parameter is used.
    """
    ixs_for_par = {}
    observable_ids = amici_model.getObservableIds()

    simulation_conditions = (
        petab_problem.get_simulation_conditions_from_measurement_df()
    )
    for condition_ix, condition in simulation_conditions.iterrows():
        df_for_condition = petab.get_rows_for_condition(
            measurement_df=petab_problem.measurement_df, condition=condition
        )

        timepoints = sorted(df_for_condition[TIME].unique().astype(float))
        timepoints_w_reps = _get_timepoints_with_replicates(
            measurement_df=df_for_condition
        )

        for time in timepoints:
            # subselect for time
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

                # update time index for observable
                if observable_ix in time_ix_for_obs_ix:
                    time_ix_for_obs_ix[observable_ix] += 1
                else:
                    time_ix_for_obs_ix[observable_ix] = time_ix_0
                time_ix = time_ix_for_obs_ix[observable_ix]

                observable_overrides = petab.split_parameter_replacement_list(
                    measurement[OBSERVABLE_PARAMETERS]
                )
                noise_overrides = petab.split_parameter_replacement_list(
                    measurement[NOISE_PARAMETERS]
                )

                # try to insert if hierarchical parameter
                for override in observable_overrides + noise_overrides:
                    if override in x_ids:
                        ixs_for_par.setdefault(override, []).append(
                            (condition_ix, time_ix, observable_ix)
                        )
    return ixs_for_par


def ix_matrices_from_arrays(
    ixs: Dict[str, List[Tuple[int, int, int]]], edatas: List[np.array]
) -> Dict[str, List[np.array]]:
    """
    Convert mapping of parameters to measurements to matrix form.

    Returns
    -------
    A dictionary mapping parameter ID to a list of Boolean matrices, one per
    simulation condition. Therein, ``True`` indicates that the respective
    parameter is used for the model output at the respective timepoint,
    observable and condition index.
    """
    ix_matrices = {
        id: [np.zeros_like(edata, dtype=bool) for edata in edatas]
        for id in ixs
    }
    for id, arr in ixs.items():
        matrices = ix_matrices[id]
        for (condition_ix, time_ix, observable_ix) in arr:
            matrices[condition_ix][time_ix, observable_ix] = True
    return ix_matrices


def _get_timepoints_with_replicates(
    measurement_df: pd.DataFrame,
) -> List[float]:
    """
    Get list of timepoints including replicate measurements.

    :param measurement_df:
        PEtab measurement table subset for a single condition.

    :return:
        Sorted list of timepoints, including multiple timepoints accounting
        for replicate measurements.
    """
    # create sorted list of all timepoints for which measurements exist
    timepoints = sorted(measurement_df[TIME].unique().astype(float))

    # find replicate numbers of time points
    timepoints_w_reps = []
    for time in timepoints:
        # subselect for time
        df_for_time = measurement_df[measurement_df.time == time]
        # rep number is maximum over rep numbers for observables
        n_reps = max(df_for_time.groupby([OBSERVABLE_ID, TIME]).size())
        # append time point n_rep times
        timepoints_w_reps.extend([time] * n_reps)

    return timepoints_w_reps
