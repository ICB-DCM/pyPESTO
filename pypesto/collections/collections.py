import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable

from ..prediction import PredictionResult
from .constants import (X_NAMES, NX, X_VECTOR, NVECTORS, VECTOR_TAGS, MODE_FUN,
                        COLLECTION_TYPE, PREDICTIONS, PREDICTION_ID, PREDICTOR,
                        PREDICTION_RESULTS, PREDICTION_ARRAY,
                        PREDICTION_SUMMARY, CollectionType as ct)


class CollectionPrediction:
    """
    A collection prediction consists of a collection, i.e., a set of parameter
    vectors and their identifiers such as a sample or an ensemble, and a
    prediction function. It can be attached to a collection-type object
    """
    def __init__(self,
                 predictor: Callable,
                 prediction_id: str = None,
                 prediction_results: Sequence[PredictionResult] = None):
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
        """
        self.predictor = predictor
        self.prediction_id = prediction_id
        self.prediction_results = prediction_results
        if prediction_results is None:
            self.prediction_results = []

        self.prediction_array = None
        self.prediction_summary = {'mean': None,
                                   'std': None,
                                   'median': None}

    def __iter__(self):
        yield PREDICTOR, self.predictor
        yield PREDICTION_ID, self.prediction_id
        yield PREDICTION_RESULTS, self.prediction_results
        yield PREDICTION_ARRAY, self.prediction_array
        yield PREDICTION_SUMMARY, self.prediction_summary


    def condense_to_array(self):
        """
        This functions reshapes the predictions results to an array
        """
        raise NotImplementedError("Condensation to an array is not yet "
                                  "implemetned.")

    def compute_summary(self):
        """
        This function computes the mean, the median, the standard deviation
        and possibly percentiles from the collection prediction results.
        """
        raise NotImplementedError("Computation of summary statistics is not "
                                  "yet implemetned.")


class Collection:
    """
    A collection is of a thin wrapper around an numpy array.
    """

    def __init__(self,
                 x_vectors: np.ndarray,
                 x_names: Sequence[str] = None,
                 vector_tags: Tuple[int, int] = None,
                 coll_type: ct = None,
                 predictions: Sequence[CollectionPrediction] = None,):
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
        coll_type:
            Type of collection: Ensemble (default) or sample
            Samples are meant to be representative, ensembles can be any
            collection of parameters
        predictions:
            List of CollectionPrediction objects
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

    def predict(self,
                predictor: Callable,
                prediction_id: str = None,
                sensi_orders: Tuple = (0,),
                mode: str = MODE_FUN,):
        """
        Convenience function to run predictions for a full ensemble
        """
        # preallocate
        prediction_results = []

        for ix in range(self.n_vectors):
            x = self.x_vectors[:,ix]
            prediction_results.append(predictor(x, sensi_orders, mode))

        return CollectionPrediction(predictor=predictor,
                                    prediction_id=prediction_id,
                                    prediction_results=prediction_results)


def read_from_csv(path: str,
                  sep: str = '\t',
                  index_col: int = 0,
                  headline_parser: Callable = None,
                  coll_type: ct = None):
    """
    function for creating an ensemble from a csv file
    """
    # get the data from the csv
    collection_df = pd.read_csv(path, sep=sep, index_col=index_col)
    # if we have a parser to make vector_tags from column names, we use it
    vector_tags = None
    if headline_parser is not None:
        vector_tags = headline_parser(list(collection_df.columns))
    # set the type of the collection
    if coll_type is None:
        coll_type = ct.ensemble

    return Collection(x_vectors=collection_df.values,
                      x_names=list(collection_df.index),
                      vector_tags=vector_tags,
                      coll_type=coll_type)
