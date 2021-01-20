"""
This is for (string) constants used in the prediction module.
"""


MODE_FUN = 'mode_fun'  # mode for function values
MODE_RES = 'mode_res'  # mode for residuals

OBSERVABLE_IDS = 'observable_ids'  # data member in PredictionConditionResult
PARAMETER_IDS = 'x_names'  # data member in PredictionConditionResult
TIMEPOINTS = 'timepoints'  # data member in PredictionConditionResult
OUTPUT = 'output'  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = 'output_sensi'  # field in the return dict of AmiciPredictor

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
