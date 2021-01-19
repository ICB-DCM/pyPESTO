import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable

from ..prediction import PredictionResult, PredictionConditionResult
from .constants import *


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
        prediction_id:
            Identifier for the predictions
        prediction_results:
            List of Prediction results
        """
        self.predictor = predictor
        self.prediction_id = prediction_id
        self.prediction_results = prediction_results
        if prediction_results is None:
            self.prediction_results = []

        self.prediction_arrays = None
        self.prediction_summary = {'mean': None,
                                   'std': None,
                                   'median': None}

    def __iter__(self):
        yield PREDICTOR, self.predictor
        yield PREDICTION_ID, self.prediction_id
        yield PREDICTION_RESULTS, self.prediction_results
        yield PREDICTION_ARRAYS, self.prediction_arrays
        yield PREDICTION_SUMMARY, self.prediction_summary

    def condense_to_arrays(self):
        """
        This functions reshapes the predictions results to an array and adds
        them as a member to the CollectionPrediction objects.
        """
        # prepare for storing results over all predictions
        output = []
        output_sensi = []
        timepoints = []

        for result in self.prediction_results:
            # stack outputs, output sensitivities and timepoints to one array
            # use first element as dummy, to see if outputs ahve been computed
            if result.conditions[0].output is not None:
                output.append(np.concatenate(
                    [cond.output for cond in result.conditions], axis=0))
            else:
                output = None

            if result.conditions[0].output_sensi is not None:
                output_sensi.append(np.concatenate(
                    [cond.output_sensi for cond in result.conditions], axis=0))
            else:
                output_sensi = None

            timepoints.append(np.concatenate(
                [cond.timepoints for cond in result.conditions], axis=0))

        # stack results in third dimension
        if output is not None:
            output = np.stack(output, axis=2)
        if output_sensi is not None:
            output_sensi = np.stack(output_sensi, axis=3)

        # formulate as dict
        self.prediction_arrays = {
            OUTPUT: output,
            OUTPUT_SENSI: output_sensi,
            TIMEPOINTS: np.stack(timepoints, axis=-1)
        }

    def compute_summary(self,
                        percentiles_list: Sequence[int] = (5, 20, 80, 95)):
        """
        This function computes the mean, the median, the standard deviation
        and possibly percentiles from the collection prediction results.
        Those summary results are added as a data member to the
        CollectionPrediction object.

        Parameters
        ----------
        percentiles_list:
            List or tuple of percent numbers for the percentiles
        """
        # check if prediction results are available
        if not self.prediction_results:
            raise ArithmeticError('Cannot compute summary statistics from '
                                  'empty prediction results.')
        n_conditions = len(self.prediction_results[0].conditions)

        def _stack_outputs(ic: int):
            """
            This function groups outputs for different parameter vectors of
            one ensemble together, if the belong to the same simulation
            condition, and stacks them in one array
            """
            # Were outputs computed
            if self.prediction_results[0].conditions[ic].output is None:
                return None
            # stack predictions
            output_list = [prediction.conditions[ic].output
                           for prediction in self.prediction_results]
            # stack into one numpy array
            return np.stack(output_list, axis=-1)

        def _stack_outputs_sensi(ic: int):
            """
            This function groups output sensitivities for different parameter
            vectors of one ensemble together, if the belong to the same
            simulation condition, and stacks them in one array
            """
            # Were output sensitivitiess computed
            if self.prediction_results[0].conditions[ic].output_sensi is None:
                return None
            # stack predictions
            output_sensi_list = [prediction.conditions[ic].output_sensi
                                 for prediction in self.prediction_results]
            # stack into one numpy array
            return np.stack(output_sensi_list, axis=-1)

        def _compute_summary(tmp_array, percentiles_list):
            """
            This function computes means, standard deviation, median, and
            requested percentiles for a set of stacked simulations
            """
            summary = {}
            summary['mean'] = np.mean(tmp_array, axis=-1)
            summary['std'] = np.std(tmp_array, axis=-1)
            summary['median'] = np.median(tmp_array, axis=-1)
            for perc in percentiles_list:
                summary[f'percentile {perc}'] = np.percentile(tmp_array,
                                                              perc, axis=-1)
            return summary

        # preallocate for results
        cond_lists = {
            'mean': [],
            'std': [],
            'median': []
        }
        for perc in percentiles_list:
            cond_lists[f'percentile {perc}'] = []

        # iterate over conditions, compute summary
        for i_cond in range(n_conditions):
            # use some short hand
            current_cond = self.prediction_results[0].conditions[i_cond]

            # create a temporary array with all the outputs needed and wanted
            tmp_output = _stack_outputs(i_cond)
            tmp_output_sensi = _stack_outputs_sensi(i_cond)

            # handle outputs
            if tmp_output is not None:
                output_summary = _compute_summary(tmp_output, percentiles_list)
            else:
                output_summary = {i_key: None for i_key in cond_lists.keys()}

            # handle output sensitivities
            if tmp_output_sensi is not None:
                output_sensi_summary = _compute_summary(tmp_output_sensi,
                                                        percentiles_list)
            else:
                output_sensi_summary = {i_key: None
                                        for i_key in cond_lists.keys()}

            # create some PredictionConditionResult to have an easier creation
            # of PredictionResults for the summaries later on
            for i_key in cond_lists.keys():
                cond_lists[i_key].append(
                    PredictionConditionResult(
                        timepoints=current_cond.timepoints,
                        output=output_summary[i_key],
                        output_sensi=output_sensi_summary[i_key],
                        observable_ids=current_cond.observable_ids
                    )
                )

        self.prediction_summary = {i_key: PredictionResult(
            conditions=cond_lists[i_key],
            condition_ids=self.prediction_results[0].condition_ids,
            comment=str(i_key))
            for i_key in cond_lists.keys()
        }


class Collection:
    """
    A collection is of a thin wrapper around an numpy array.
    """

    def __init__(self,
                 x_vectors: np.ndarray,
                 x_names: Sequence[str] = None,
                 vector_tags: Tuple[int, int] = None,
                 coll_type: CollectionType = None,
                 predictions: Sequence[CollectionPrediction] = None, ):
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
        self.n_x = x_vectors.shape[0]
        self.n_vectors = x_vectors.shape[1]
        self.vector_tags = vector_tags

        # handle parameter names
        if x_names is not None:
            self.x_names = x_names
        else:
            self.x_names = [f'x_{ix}' for ix in range(self.n_x)]

        # Do we have a representative sample or just random ensemble?
        self.coll_type = CollectionType.ensemble
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
                mode: str = MODE_FUN, ):
        """
        Convenience function to run predictions for a full ensemble:
        User needs to hand over a predictor function and settings, then all
        results are grouped as CollectionPrediction for the whole ensemble

        Parameters
        ----------
        predictor:
            Prediction function, e.g., an AmiciPredictor
        prediction_id:
            Identifier for the predictions
        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad.
        mode:
            Whether to compute function values or residuals.

        Returns
        -------
        result:
            CollectionPrediction of the collection object for the predictor
            function
        """
        # preallocate
        prediction_results = []

        for ix in range(self.n_vectors):
            x = self.x_vectors[:, ix]
            prediction_results.append(predictor(x, sensi_orders, mode))

        return CollectionPrediction(predictor=predictor,
                                    prediction_id=prediction_id,
                                    prediction_results=prediction_results)


def read_from_csv(path: str,
                  sep: str = '\t',
                  index_col: int = 0,
                  headline_parser: Callable = None,
                  coll_type: CollectionType = None):
    """
    function for creating an ensemble from a csv file

    Parameters
    ----------
    path:
        path to csv file to read in parameter collection/ensemble
    sep:
        separator in csv file
    index_col:
        index column in csv file
    headline_parser:
        A function which reads in the headline of the csv file and converts it
        into vector_tags (see constructor of Collection for more details)
    coll_type:
        Collection type: representative sample or random ensemble

    Returns
    -------
    result:
        Collection object of parameter vectors
    """
    # get the data from the csv
    collection_df = pd.read_csv(path, sep=sep, index_col=index_col)
    # if we have a parser to make vector_tags from column names, we use it
    vector_tags = None
    if headline_parser is not None:
        vector_tags = headline_parser(list(collection_df.columns))
    # set the type of the collection
    if coll_type is None:
        coll_type = CollectionType.ensemble

    return Collection(x_vectors=collection_df.values,
                      x_names=list(collection_df.index),
                      vector_tags=vector_tags,
                      coll_type=coll_type)
