"""Inner optimization problem in hierarchical optimization."""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from ..C import LIN, LOG, LOG10
from .base_parameter import InnerParameter

try:
    import amici
    import petab.v1 as petab
    from petab.v1.C import OBSERVABLE_ID, TIME
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

    def __init__(self, xs: list[InnerParameter], data: list[np.ndarray]):
        self.xs: dict[str, InnerParameter] = {
            x.inner_parameter_id: x for x in xs
        }
        self.data = copy.deepcopy(data)

        # create the joint mask of all inner problem parameters
        self.data_mask = [
            np.zeros_like(cond_data, dtype=bool) for cond_data in data
        ]
        for x in xs:
            for condition_ix, cond_mask in enumerate(self.data_mask):
                cond_mask[x.ixs[condition_ix]] = True

        # mask the data: a inner problem is aware of only the data it uses
        for i in range(len(self.data)):
            self.data[i][~self.data_mask[i]] = np.nan

        logger.debug(f"Created InnerProblem with ids {self.get_x_ids()}")

        if self.is_empty():
            raise ValueError(
                "There are no parameters in the inner problem of hierarchical "
                "optimization."
            )

    @staticmethod
    def from_petab_amici(
        petab_problem: petab.Problem,
        amici_model: amici.Model,
        edatas: list[amici.ExpData],
    ) -> InnerProblem:
        """Create an InnerProblem from a PEtab problem and AMICI objects."""

    def get_x_ids(self) -> list[str]:
        """Get IDs of inner parameters."""
        return list(self.xs.keys())

    def get_interpretable_x_ids(self) -> list[str]:
        """Get IDs of interpretable inner parameters.

        Interpretable parameters need to be easily interpretable by the user.
        Examples are scaling factors, offsets, or noise parameters. An example
        of non-interpretable inner parameters is the spline heights of spline
        approximation for semiquantitative data. It is challenging to interpret
        the meaning of these parameters based solely on their value.
        """
        return list(self.xs.keys())

    def get_interpretable_x_scales(self) -> list[str]:
        """Get scales of interpretable inner parameters."""
        return [x.scale for x in self.xs.values()]

    def get_xs_for_type(
        self, inner_parameter_type: str
    ) -> list[InnerParameter]:
        """Get inner parameters of the given type."""
        return [
            x
            for x in self.xs.values()
            if x.inner_parameter_type == inner_parameter_type
        ]

    def get_dummy_values(self, scaled: bool) -> dict[str, float]:
        """
        Get dummy parameter values.

        Get parameter values to be used for simulation before their optimal
        values are computed.

        Parameters
        ----------
        scaled:
            Whether the parameters should be returned on parameter scale (``True``)
            or on linear scale (``False``).
        """
        return {
            x.inner_parameter_id: (
                scale_value(x.dummy_value, x.scale)
                if scaled
                else x.dummy_value
            )
            for x in self.xs.values()
        }

    def get_for_id(self, inner_parameter_id: str) -> InnerParameter:
        """Get InnerParameter for the given parameter ID."""
        try:
            return self.xs[inner_parameter_id]
        except KeyError:
            raise KeyError(
                f"Cannot find parameter with id {inner_parameter_id}."
            ) from None

    def is_empty(self) -> bool:
        """Check for emptiness.

        Returns
        -------
        ``True`` if there aren't any parameters associated with this problem,
        ``False`` otherwise.
        """
        return len(self.xs) == 0

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds of inner parameters."""
        lb = np.asarray([x.lb for x in self.xs.values()])
        ub = np.asarray([x.ub for x in self.xs.values()])
        return lb, ub

    def get_interpretable_x_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bounds of interpretable inner parameters."""
        interpretable_x_ids = self.get_interpretable_x_ids()
        lb = np.asarray(
            [
                x.lb
                for x in self.xs.values()
                if x.inner_parameter_id in interpretable_x_ids
            ]
        )
        ub = np.asarray(
            [
                x.ub
                for x in self.xs.values()
                if x.inner_parameter_id in interpretable_x_ids
            ]
        )
        return lb, ub


class AmiciInnerProblem(InnerProblem):
    """
    Inner optimization problem in hierarchical optimization.

    For use with AMICI objects.

    Attributes
    ----------
    edatas:
        AMICI ``ExpDataView``s for each simulation condition.
    """

    def __init__(self, edatas: list[amici.ExpData], **kwargs):
        super().__init__(**kwargs)

    def check_edatas(self, edatas: list[amici.ExpData]) -> bool:
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
            amici.numpy.ExpDataView(edata)["observedData"] for edata in edatas
        ]

        # Mask the data using the inner problem mask. This is necessary
        # because the inner problem is aware of only the data it uses.
        for i in range(len(data)):
            data[i][~self.data_mask[i]] = np.nan

        if len(self.data) != len(data):
            return False

        for data0, data1 in zip(self.data, data):
            if not np.array_equal(data0, data1, equal_nan=True):
                return False

        return True


def scale_value_dict(
    dct: dict[str, float], problem: InnerProblem
) -> dict[str, float]:
    """Scale a value dictionary."""
    scaled_dct = {}
    for key, val in dct.items():
        x = problem.get_for_id(key)
        scaled_dct[key] = scale_value(val, x.scale)
    return scaled_dct


def scale_value(val: float | np.array, scale: str) -> float | np.array:
    """Scale a single value."""
    if scale == LIN:
        return val
    if scale == LOG:
        return np.log(val)
    if scale == LOG10:
        return np.log10(val)
    raise ValueError(f"Scale {scale} not recognized.")


def scale_back_value_dict(
    dct: dict[str, float], problem: InnerProblem
) -> dict[str, float]:
    """Scale back a value dictionary."""
    scaled_dct = {}
    for key, val in dct.items():
        x = problem.get_for_id(key)
        scaled_dct[key] = scale_back_value(val, x.scale)
    return scaled_dct


def scale_back_value(val: float | np.array, scale: str) -> float | np.array:
    """Scale back a single value."""
    if scale == LIN:
        return val
    if scale == LOG:
        return np.exp(val)
    if scale == LOG10:
        return 10**val
    raise ValueError(f"Scale {scale} not recognized.")


def ix_matrices_from_arrays(
    ixs: dict[str, list[tuple[int, int, int]]], edatas: list[np.array]
) -> dict[str, list[np.array]]:
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
) -> list[float]:
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
