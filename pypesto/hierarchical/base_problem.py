"""Inner optimization problem in hierarchical optimization."""
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from .base_parameter import InnerParameter

try:
    import amici
    import petab
    from petab.C import OBSERVABLE_ID, TIME
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

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Get bounds of inner parameters."""
        lb = [x.lb for x in self.xs.values()]
        ub = [x.ub for x in self.xs.values()]
        return lb, ub


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
