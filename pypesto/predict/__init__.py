"""
Prediction
==========

Generate predictions from simulations with specified parameter vectors, with
optional post-processing.
"""


from .amici_predictor import AmiciPredictor
from .result import (
    PredictionResult,
    PredictionConditionResult,
)
from .task import PredictorTask
