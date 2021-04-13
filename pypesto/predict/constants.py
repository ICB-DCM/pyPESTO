"""Constants used in the `pypesto.predict` module."""


MODE_FUN = 'mode_fun'  # mode for function values
MODE_RES = 'mode_res'  # mode for residuals

OUTPUT_IDS = 'output_ids'  # data member in PredictionConditionResult
PARAMETER_IDS = 'x_names'  # data member in PredictionConditionResult
CONDITION_IDS = 'condition_ids'
TIMEPOINTS = 'timepoints'  # data member in PredictionConditionResult
OUTPUT = 'output'  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = 'output_sensi'  # field in the return dict of AmiciPredictor

# separator in the conditions_ids betweeen preequilibration and simulation
# condition
CONDITION_SEP = '::'

RDATAS = 'rdatas'  # return field of call to pypesto objective
AMICI_T = 't'  # return field in amici simulation result
AMICI_X = 'x'  # return field in amici simulation result
AMICI_SX = 'sx'  # return field in amici simulation result
AMICI_Y = 'y'  # return field in amici simulation result
AMICI_SY = 'sy'  # return field in amici simulation result
AMICI_STATUS = 'status'  # return field in amici simulation result

CSV = 'csv'  # return file format
H5 = 'h5'  # return file format
TIME = 'time'  # column name in returned csv

CONDITION = 'condition'
CONDITION_IDS = 'condition_ids'


def get_condition_label(condition_id: str) -> str:
    """Convert a condition ID to a label.

    Labels for conditions are used at different locations (e.g. ensemble
    prediction code, and visualization code). This method ensures that the same
    condition is labeled identically everywhere.

    Parameters
    ----------
    condition_id:
        The condition ID that will be used to generate a label.

    Returns
    -------
    The condition label.
    """
    return f'condition_{condition_id}'
