"""Inner optimization problem in hierarchical optimization."""
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from ..C import PARAMETER_TYPE, InnerParameterType
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


class InnerProblem:
    """
    Inner optimization problem in hierarchical optimization.

    Attributes
    ----------
    xs:
        Mapping of (inner) parameter ID to ``InnerParameters``.
    data:
        Measurement data. One matrix (`num_timepoints` x `num_observables`)
        per simulation condition. Missing observations as NaN.
    """

    def __init__(self, xs: List[InnerParameter], data: List[np.ndarray]):
        self.xs: Dict[str, InnerParameter] = {
            x.inner_parameter_id: x for x in xs
        }
        self.data = data

        logger.debug(f"Created InnerProblem with ids {self.get_x_ids()}")

        if self.is_empty():
            raise ValueError(
                'There are no parameters in the inner problem of hierarchical '
                'optimization.'
            )

    @staticmethod
    def from_petab_amici(
        petab_problem: 'petab.Problem',
        amici_model: 'amici.Model',
        edatas: List['amici.ExpData'],
    ) -> 'InnerProblem':
        """Create an InnerProblem from a PEtab problem and AMICI objects."""
        return inner_problem_from_petab_problem(
            petab_problem, amici_model, edatas
        )

    def get_x_ids(self) -> List[str]:
        """Get IDs of inner parameters."""
        return list(self.xs.keys())

    def get_xs_for_type(
        self, inner_parameter_type: str
    ) -> List[InnerParameter]:
        """Get inner parameters of the given type."""
        return [
            x
            for x in self.xs.values()
            if x.inner_parameter_type == inner_parameter_type
        ]

    def get_dummy_values(self, scaled: bool) -> Dict[str, float]:
        """
        Get dummy parameter values.

        Get parameter values to be used for simulation before their optimal
        values are computed.

        Parameters
        ----------
        scaled:
            Whether the parameters should be returned on parameter or linear
            scale.
        """
        return {
            x.inner_parameter_id: scale_value(x.dummy_value, x.scale)
            if scaled
            else x.dummy_value
            for x in self.xs.values()
        }

    def get_for_id(self, inner_parameter_id: str) -> InnerParameter:
        """Get InnerParameter for the given parameter ID."""
        try:
            return self.xs[inner_parameter_id]
        except KeyError:
            raise KeyError(f"Cannot find parameter with id {id}.")

    def is_empty(self) -> bool:
        """Check for emptiness.

        Returns
        -------
        ``True`` if there aren't any parameters associated with this problem,
        ``False`` otherwise.
        """
        return len(self.xs) == 0


class AmiciInnerProblem(InnerProblem):
    """
    Inner optimization problem in hierarchical optimization.

    For use with AMICI objects.

    Attributes
    ----------
    edataviews:
        AMICI ``ExpDataView``s for each simulation condition.
    """

    def __init__(self, edatas: List[amici.ExpData], **kwargs):
        super().__init__(**kwargs)

    def check_edatas(self, edatas: List[amici.ExpData]) -> bool:
        """Check for consistency in data.

        Currently only checks for the actual data values. e.g., timepoints are
        not compared.

        Parameters
        ----------
        edatas:
            A data set. Will be checked against the data set provided to the
            constructor.

        Returns
        -------
        Whether the data sets are consistent.
        """
        # TODO replace but edata1==edata2 once this makes it into amici
        #  https://github.com/AMICI-dev/AMICI/issues/1880
        data = [
            amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas
        ]

        if len(self.data) != len(data):
            return False

        for data0, data1 in zip(self.data, data):
            if not np.array_equal(data0, data1, equal_nan=True):
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

    x_ids = [x.inner_parameter_id for x in inner_parameters]

    # used indices for all measurement specific parameters
    ixs = ixs_for_measurement_specific_parameters(
        petab_problem, amici_model, x_ids
    )

    # transform experimental data
    data = [amici.numpy.ExpDataView(edata)['observedData'] for edata in edatas]

    # matrixify
    ix_matrices = ix_matrices_from_arrays(ixs, data)

    # assign matrices
    for par in inner_parameters:
        par.ixs = ix_matrices[par.inner_parameter_id]

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
        if (
            (InnerParameterType.SCALING in types)
            and (InnerParameterType.OFFSET in types)
        )
        for par in group
    }

    for par in inner_parameters:
        if par.inner_parameter_type not in [
            InnerParameterType.SCALING,
            InnerParameterType.OFFSET,
        ]:
            continue
        if par.inner_parameter_id in coupled_pars:
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

    for col in (PARAMETER_TYPE,):
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
                inner_parameter_id=row[PARAMETER_ID],
                inner_parameter_type=row[PARAMETER_TYPE],
                scale=row[PARAMETER_SCALE],
                lb=row[LOWER_BOUND],
                ub=row[UPPER_BOUND],
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
    A dictionary mapping parameter ID to a List of
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

                observable_overrides = petab.split_parameter_replacement_list(
                    measurement.get(OBSERVABLE_PARAMETERS, None)
                )
                noise_overrides = petab.split_parameter_replacement_list(
                    measurement.get(NOISE_PARAMETERS, None)
                )

                # try to insert if hierarchical parameter
                for override in observable_overrides + noise_overrides:
                    if override in x_ids:
                        ixs_for_par.setdefault(override, []).append(
                            (condition_ix, time_w_reps_ix, observable_ix)
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
        for condition_ix, time_ix, observable_ix in arr:
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
