import numpy as np
from typing import Sequence, Tuple, Callable

from ..prediction import PredictionResult
from .constants import (X_NAMES, NX, X_VECTOR, NVECTORS, VECTOR_TAGS,
                        COLLECTION_TYPE, PREDICTIONS, PREDICTOR, PREDICTION_ID,
                        PREDICTION_RESULTS, PREDICTION_ARRAY,
                        CollectionType as ct)


class PredictionCollection:
    """
    A prediction collection consists of a collection, i.e., a set of parameter
    vectors and their identifiers such as a sample or an ensemble, and a
    prediction function. It can be attached to a collection-type object
    """
    def __init__(self,
                 predictor: Callable,
                 prediction_id: str = None,
                 prediction_results: Sequence[PredictionResult] = None,
                 prediction_array: np.ndarray = None):
        """
        Constructor.

        Parameters
        ----------
        predictor:
            Prediction function, e.g., an AmiciPredictor
        prediction_ids:
            Identifiers for the predictions
        prediction_results:
            List of Prediction results
        prediction_array:
            prediction results condensed to a numpy array
        """
        self.predictor = predictor
        self.prediction_id = prediction_id
        self.prediction_results = prediction_results
        if prediction_results is None:
            self.prediction_results = []

        self.prediction_array = prediction_array

    def __iter__(self):
        yield PREDICTOR, self.predictor
        yield PREDICTION_ID, self.prediction_id
        yield PREDICTION_RESULTS, self.prediction_results
        yield PREDICTION_ARRAY, self.prediction_array


class Collection:
    """
    A collection is of a thin wrapper around an numpy array.
    """

    def __init__(self,
                 x_vectors: np.ndarray,
                 x_names: Sequence[str] = None,
                 vector_tags: Tuple[int, int] = None,
                 coll_type: ct = None,
                 predictions: Sequence[PredictionCollection] = None,):
        """
        Constructor.

        Parameters
        ----------
        x_vectors:
            parameter vectors of the collection, in the format
            n_parameter x n_vectors
        x_names:
            Nmes or identifiers of the parameters
        vector_tags:
            Additional tag, which adds information about the the parameter
            vectors of the form (optimization_run, optimization_step) if the
            collection is created from an optimization result or
            (sampling_chain, sampling_step) if the collection is created from a
            sampling result.
        """

        # handle parameter vectors and sizes
        self.x_vectors = x_vectors
        self.n_x = x_vectors.size[0]
        self.n_vectors = x_vectors.size[1]
        self.vector_tags = vector_tags

        # handle parameter names
        if x_names is not None:
            self.x_names = x_names
        else:
            self.x_names = [f'x_{ix}' for ix in range(self.n_x)]

        # Do we have a representative sample or just random ensemble?
        self.coll_type = ct.ensemble
        if coll_type is not None:
            self.coll_type = coll_type

        # Do we have predictions for this ensemble?
        self.predictions = []
        if predictions is not None:
            self.predictions = predictions

    def __iter__(self):
        yield X_VECTOR, self.x_vectors
        yield NX, self.n_x
        yield X_NAMES, self.x_names
        yield NVECTORS, self.n_vectors
        yield VECTOR_TAGS, self.vector_tags
        yield COLLECTION_TYPE, self.coll_type
        yield PREDICTIONS, self.predictions
