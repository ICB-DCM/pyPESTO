"""
This is for (string) constants used in the collections module.
"""


from enum import Enum


MODE_FUN = 'mode_fun'  # mode for function values

OBSERVABLE_IDS = 'observable_ids'  # data member in PredictionConditionResult
PARAMETER_IDS = 'parameter_ids'  # data member in PredictionConditionResult
TIMEPOINTS = 'timepoints'  # data member in PredictionConditionResult
OUTPUT = 'output'  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = 'output_sensi'  # field in the return dict of AmiciPredictor

PREDICTOR = 'predictor'
PREDICTION_ID = 'prediction_id'
PREDICTION_RESULTS = 'predction_results'
PREDICTION_ARRAYS = 'prediction_arrays'
PREDICTION_SUMMARY = 'prediction_summary'

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
COLLECTION_TYPE = 'coll_type'
PREDICTIONS = 'predictions'

LOWER_BOUND = 'lower_bound'
UPPER_BOUND = 'upper_bound'

CSV = 'csv'  # return file format
H5 = 'h5'  # return file format
TIME = 'time'  # column name in returned csv

COLOR_HIT_BOTH_BOUNDS = [0.6, 0., 0., 0.9]
COLOR_HIT_ONE_BOUND = [0.95, 0.6, 0., 0.9]
COLOR_HIT_NO_BOUNDS = [0., 0.8, 0., 0.9]

class CollectionType(Enum):
    ensemble = 1
    sample = 2
    unprocessed_chain = 3
