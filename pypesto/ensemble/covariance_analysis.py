import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable, Dict

from ..prediction import PredictionResult, PredictionConditionResult
from ..ensemble import Ensemble
from .constants import (PREDICTOR, PREDICTION_ID, PREDICTION_RESULTS,
                        PREDICTION_ARRAYS, PREDICTION_SUMMARY, OUTPUT,
                        OUTPUT_SENSI, TIMEPOINTS, X_VECTOR, NX, X_NAMES,
                        NVECTORS, VECTOR_TAGS, PREDICTIONS, MODE_FUN,
                        EnsembleType, ENSEMBLE_TYPE, MEAN, MEDIAN,
                        STANDARD_DEVIATION, PERCENTILE, SUMMARY, LOWER_BOUND,
                        UPPER_BOUND)


def get_covariance_matrix_parameters(ens: Ensemble):
    # bla
    covariance = np.cov(ens.x_vectors.transpose())

def get_covariance_matrix_predictions():
    raise Exception('This is not yet integrated')
    pass

def get_spectral_decomposition_parameters():
    pass

def get_spectral_decomposition_predictions():
    pass

def _get_covariance_matrix_lowlevel():
    pass

def _get_spectral_decomposition_lowlevel():
    pass
