"""
This is for (string) constants used in the collections module.
"""


from enum import Enum


OBSERVABLE_IDS = 'observable_ids'  # data member in PredictionConditionResult
PARAMETER_IDS = 'parameter_ids'  # data member in PredictionConditionResult
TIMEPOINTS = 'timepoints'  # data member in PredictionConditionResult
OUTPUT = 'output'  # field in the return dict of AmiciPredictor
OUTPUT_SENSI = 'output_sensi'  # field in the return dict of AmiciPredictor

PREDICTOR = 'predictor'
PREDICTION_ID = 'prediction_id'
PREDICTION_RESULTS = 'predction_results'
PREDICTION_ARRAY = 'rediction_array'

X_NAMES = 'x_names'
NX = 'n_x'
X_VECTOR = 'x_vectors'
NVECTORS = 'n_vectors'
VECTOR_TAGS = 'vector_tags'
COLLECTION_TYPE = 'coll_type'
PREDICTIONS = 'predictions'

CSV = 'csv'  # return file format
H5 = 'h5'  # return file format
TIME = 'time'  # column name in returned csv

class CollectionType(Enum):
    ensemble = 1
    sample = 2
