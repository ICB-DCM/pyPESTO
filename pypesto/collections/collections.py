import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable, Dict

from ..prediction import PredictionResult, PredictionConditionResult
from .constants import (PREDICTOR, PREDICTION_ID, PREDICTION_RESULTS,
                        PREDICTION_ARRAYS, PREDICTION_SUMMARY, OUTPUT,
                        OUTPUT_SENSI, TIMEPOINTS, X_VECTOR, NX, X_NAMES,
                        NVECTORS, VECTOR_TAGS, PREDICTIONS, MODE_FUN,
                        CollectionType, COLLECTION_TYPE, MEAN, MEDIAN,
                        STANDARD_DEVIATION, PERCENTILE, SUMMARY, LOWER_BOUND,
                        UPPER_BOUND)


class CollectionPrediction:
    """
    A collection prediction consists of a collection, i.e., a set of parameter
    vectors and their identifiers such as a sample or an ensemble, and a
    prediction function. It can be attached to a collection-type object
    """

    def __init__(self,
                 predictor: Callable,
                 prediction_id: str = None,
                 prediction_results: Sequence[PredictionResult] = None,
                 lower_bound: Sequence[np.ndarray] = None,
                 upper_bound: Sequence[np.ndarray] = None):
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
        lower_bound:
            Array of potential lower bounds for the predictions, should have the
            same shape as the output of the predictions, i.e., a list of numpy
            array (one list entry per condition), with the arrays having the
            shape of n_timepoints x n_outputs for each condition.
        upper_bound:
            array of potential upper bounds for the parameters
        """
        self.predictor = predictor
        self.prediction_id = prediction_id
        self.prediction_results = prediction_results
        if prediction_results is None:
            self.prediction_results = []

        # handle bounds
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.prediction_arrays = None
        self.prediction_summary = {MEAN: None,
                                   STANDARD_DEVIATION: None,
                                   MEDIAN: None}

    def __iter__(self):
        yield PREDICTOR, self.predictor
        yield PREDICTION_ID, self.prediction_id
        yield PREDICTION_RESULTS, self.prediction_results
        yield PREDICTION_ARRAYS, self.prediction_arrays
        yield PREDICTION_SUMMARY, {i_key: dict(self.prediction_summary[i_key])
                                   for i_key in self.prediction_summary.keys()}
        yield LOWER_BOUND, self.lower_bound
        yield UPPER_BOUND, self.upper_bound

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
            # use first element as dummy, to see if outputs have been computed
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
                        percentiles_list: Sequence[int] = (5, 20, 80, 95)
                        ) -> Dict:
        """
        This function computes the mean, the median, the standard deviation
        and possibly percentiles from the collection prediction results.
        Those summary results are added as a data member to the
        CollectionPrediction object.

        Parameters
        ----------
        percentiles_list:
            List or tuple of percent numbers for the percentiles

        Returns
        -------
        summary:
            dictionary of predictions results with the keys mean, std, median,
            percentiles, ...
        """
        # check if prediction results are available
        if not self.prediction_results:
            raise ArithmeticError('Cannot compute summary statistics from '
                                  'empty prediction results.')
        n_conditions = len(self.prediction_results[0].conditions)

        def _stack_outputs(ic: int):
            """
            This function groups outputs for different parameter vectors of
            one ensemble together, if they belong to the same simulation
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
            summary[MEAN] = np.mean(tmp_array, axis=-1)
            summary[STANDARD_DEVIATION] = np.std(tmp_array, axis=-1)
            summary[MEDIAN] = np.median(tmp_array, axis=-1)
            for perc in percentiles_list:
                summary[f'{PERCENTILE} {perc}'] = np.percentile(tmp_array,
                                                                perc, axis=-1)
            return summary

        # preallocate for results
        cond_lists = {MEAN: [], STANDARD_DEVIATION: [], MEDIAN: []}
        for perc in percentiles_list:
            cond_lists[f'{PERCENTILE} {perc}'] = []

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

        # also return the object
        return self.prediction_summary


class Collection:
    """
    A collection is a thin wrapper around an numpy array.
    """

    def __init__(self,
                 x_vectors: np.ndarray,
                 x_names: Sequence[str] = None,
                 vector_tags: Sequence[Tuple[int, int]] = None,
                 coll_type: CollectionType = None,
                 predictions: Sequence[CollectionPrediction] = None,
                 lower_bound: np.ndarray = None,
                 upper_bound: np.ndarray = None):
        """
        Constructor.

        Parameters
        ----------
        x_vectors:
            parameter vectors of the collection, in the format
            n_parameter x n_vectors
        x_names:
            Names or identifiers of the parameters
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
        lower_bound:
            array of potential lower bounds for the parameters
        upper_bound:
            array of potential upper bounds for the parameters
        """

        # handle parameter vectors and sizes
        self.x_vectors = x_vectors
        self.n_x = x_vectors.shape[0]
        self.n_vectors = x_vectors.shape[1]
        self.vector_tags = vector_tags
        self.summary = None

        # store bounds
        self.lower_bound = np.full((self.n_x,), np.nan)
        if lower_bound is not None:
            self.lower_bound = lower_bound
        self.upper_bound = np.full((self.n_x,), np.nan)
        if upper_bound is not None:
            self.upper_bound = upper_bound

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
        yield SUMMARY, self.summary
        yield LOWER_BOUND, self.lower_bound
        yield UPPER_BOUND, self.upper_bound

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

    def compute_summary(self,
                        percentiles_list: Sequence[int] = (5, 20, 80, 95)):
        """
        This function computes the mean, the median, the standard deviation
        and possibly percentiles for the parameters of the collection.
        Those summary results are added as a data member to the
        CollectionPrediction object.

        Parameters
        ----------
        percentiles_list:
            List or tuple of percent numbers for the percentiles

        Returns
        -------
        summary:
            Dict with mean, std, median, and percentiles of parameter vectors
        """
        # compute summaries based on parameters
        summary = {MEAN: np.mean(self.x_vectors, axis=1),
                   STANDARD_DEVIATION: np.std(self.x_vectors, axis=1),
                   MEDIAN: np.median(self.x_vectors, axis=1)}
        for perc in percentiles_list:
            summary[f'{PERCENTILE} {perc}'] = np.percentile(self.x_vectors,
                                                            perc, axis=1)
        # store and return results
        self.summary = summary
        return summary

    def check_identifiability(self) -> pd.DataFrame:
        """
        This functions uses ensemble mean and standard deviation to assess
        (in a rudimentary way) whether or not parameters are identifiable.
        It returns a dataframe with tuples, which specify whether or not the
        lower and the upper bound are violated

        Returns
        -------
        parameter_identifiability:
            DataFrame indicating parameter identifiability based on mean
            plus/minus standard deviations and parameter bounds
        """
        # first check if summary was computed. If not, do so
        if self.summary is None:
            self.compute_summary()

        # check identifiability for each parameter
        parameter_identifiability = []
        for ix, x_name in enumerate(self.x_names):
            # define some short hands
            lb = self.lower_bound[ix]
            ub = self.upper_bound[ix]
            mean = self.summary[MEAN][ix]
            std = self.summary[STANDARD_DEVIATION][ix]
            median = self.summary[MEAN][ix]

            # create dict of identifiability
            parameter_identifiability.append({
                'parameterId': x_name,
                'lowerBound': lb,
                'upperBound': ub,
                'collection_mean': mean,
                'collection_std': std,
                'collection_median': median,
                'within lb: 1 std': lb < mean - std,
                'within ub: 1 std': ub > mean + std,
                'within lb: 2 std': lb < mean - 2 * std,
                'within ub: 2 std': ub > mean + 2 * std,
                'within lb: 3 std': lb < mean - 3 * std,
                'within ub: 3 std': ub > mean + 3 * std,
            })

        # create DataFrame
        parameter_identifiability = pd.DataFrame(parameter_identifiability)
        parameter_identifiability.index = \
            parameter_identifiability['parameterId']

        return parameter_identifiability


def read_from_csv(path: str,
                  sep: str = '\t',
                  index_col: int = 0,
                  headline_parser: Callable = None,
                  coll_type: CollectionType = None,
                  lower_bound: np.ndarray = None,
                  upper_bound: np.ndarray = None):
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
    lower_bound:
        array of potential lower bounds for the parameters
    upper_bound:
        array of potential upper bounds for the parameters

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
                      coll_type=coll_type,
                      lower_bound=lower_bound,
                      upper_bound=upper_bound)
