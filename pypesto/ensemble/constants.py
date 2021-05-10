"""
This is for (string) constants used in the ensemble module.
"""


from enum import Enum
from typing import Union

from ..predict.constants import (  # noqa: F401
    OUTPUT,
    OUTPUT_IDS,
    OUTPUT_SENSI,
    PARAMETER_IDS,
    TIMEPOINTS,
)


MODE_FUN = 'mode_fun'  # mode for function values

PREDICTOR = 'predictor'
PREDICTION_ID = 'prediction_id'
PREDICTION_RESULTS = 'prediction_results'
PREDICTION_ARRAYS = 'prediction_arrays'
PREDICTION_SUMMARY = 'prediction_summary'

HISTORY = 'history'
OPTIMIZE = 'optimize'
SAMPLE = 'sample'

MEAN = 'mean'
MEDIAN = 'median'
STANDARD_DEVIATION = 'std'
PERCENTILE = 'percentile'
SUMMARY = 'summary'

X_NAMES = 'x_names'
NX = 'n_x'
X_VECTOR = 'x_vectors'
NVECTORS = 'n_vectors'
VECTOR_TAGS = 'vector_tags'
ENSEMBLE_TYPE = 'ensemble_type'
PREDICTIONS = 'predictions'

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'
PREEQUILIBRATION_CONDITION_ID = 'preequilibrationConditionId'
SIMULATION_CONDITION_ID = 'simulationConditionId'

CSV = 'csv'  # return file format
H5 = 'h5'  # return file format
TIME = 'time'  # column name in returned csv

COLOR_HIT_BOTH_BOUNDS = [0.6, 0., 0., 0.9]
COLOR_HIT_ONE_BOUND = [0.95, 0.6, 0., 0.9]
COLOR_HIT_NO_BOUNDS = [0., 0.8, 0., 0.9]


class EnsembleType(Enum):
    ensemble = 1
    sample = 2
    unprocessed_chain = 3


def get_percentile_label(percentile: Union[float, int, str]) -> str:
    """Convert a percentile to a label.

    Labels for percentiles are used at different locations (e.g. ensemble
    prediction code, and visualization code). This method ensures that the same
    percentile is labeled identically everywhere.

    The percentile is rounded to two decimal places in the label representation
    if it is specified to more decimal places. This is for readability in
    plotting routines, and to avoid float to string conversion issues related
    to float precision.

    Parameters
    ----------
    percentile:
        The percentile value that will be used to generate a label.

    Returns
    -------
    The label of the (possibly rounded) percentile.
    """
    if isinstance(percentile, str):
        percentile = float(percentile)
        if percentile == round(percentile):
            percentile = round(percentile)
    if isinstance(percentile, float):
        percentile_str = f'{percentile:.2f}'
        # Add `...` to the label if the percentile value changed after rounding
        if float(percentile_str) != percentile:
            percentile_str += '...'
        percentile = percentile_str
    return f'{PERCENTILE} {percentile}'
