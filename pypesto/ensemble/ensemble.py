import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable, Dict

from ..prediction import PredictionResult, PredictionConditionResult
from .constants import (PREDICTOR, PREDICTION_ID, PREDICTION_RESULTS,
                        PREDICTION_ARRAYS, PREDICTION_SUMMARY, OUTPUT,
                        OUTPUT_SENSI, TIMEPOINTS, X_VECTOR, NX, X_NAMES,
                        NVECTORS, VECTOR_TAGS, PREDICTIONS, MODE_FUN,
                        EnsembleType, ENSEMBLE_TYPE, MEAN, MEDIAN,
                        STANDARD_DEVIATION, PERCENTILE, SUMMARY, LOWER_BOUND,
                        UPPER_BOUND)


class EnsemblePrediction:
    """
    A ensemble prediction consists of an ensemble, i.e., a set of parameter
    vectors and their identifiers such as a sample, and a prediction function.
    It can be attached to a ensemble-type object
    """

    def __init__(self,
                 predictor: Callable[[Sequence], PredictionResult],
                 prediction_id: str = None,
                 prediction_results: Sequence[PredictionResult] = None,
                 lower_bound: Sequence[np.ndarray] = None,
                 upper_bound: Sequence[np.ndarray] = None):
        """
        Constructor.

        Parameters
        ----------
        predictor:
            Prediction function, e.g., an AmiciPredictor, which takes a
            parameter vector as input and outputs a PredictionResult object
        prediction_id:
            Identifier for the predictions
        prediction_results:
            List of Prediction results
        lower_bound:
            Array of potential lower bounds for the predictions, should have
            the same shape as the output of the predictions, i.e., a list of
            numpy array (one list entry per condition), with the arrays having
            the shape of n_timepoints x n_outputs for each condition.
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
        """
        __iter__ makes the instances of the class iterable objects, allowing to
        apply functions such as __dict__ to them.
        """
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
        them as a member to the EnsemblePrediction objects. It's meant to be
        used only if all conditions of a prediction have the same observables,
        as this is often the case for large-scale data sets taken from online
        databases or similar.
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
        Compute the mean, the median, the standard deviation and possibly
        percentiles from the ensemble prediction results. Those summary results
        are added as a data member to the EnsemblePrediction object.

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
            Group outputs for different parameter vectors of one ensemble
            together, if they belong to the same simulation condition, and
            stacks them in one array
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
            Group output sensitivities for different parameter vectors of one
            ensemble together, if the belong to the same simulation condition,
            and stacks them in one array
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
            Computes means, standard deviation, median, and requested
            percentiles for a set of stacked simulations
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


class Ensemble:
    """
    A ensemble is a wrapper around an numpy array. It comes with some
    convenience functionality: It allows to map parameter values via
    identifiers to the correct parameters, it allows to compute summaries of
    the parameter vectors (mean, standard deviation, median, percentiles) more
    easily, and it can store predictions made by pyPESTO, such that the
    parameter ensemble and the predictions are linked to each other.
    """

    def __init__(self,
                 x_vectors: np.ndarray,
                 x_names: Sequence[str] = None,
                 vector_tags: Sequence[Tuple[int, int]] = None,
                 ensemble_type: EnsembleType = None,
                 predictions: Sequence[EnsemblePrediction] = None,
                 lower_bound: np.ndarray = None,
                 upper_bound: np.ndarray = None):
        """
        Constructor.

        Parameters
        ----------
        x_vectors:
            parameter vectors of the ensemble, in the format
            n_parameter x n_vectors
        x_names:
            Names or identifiers of the parameters
        vector_tags:
            Additional tag, which adds information about the the parameter
            vectors of the form (optimization_run, optimization_step) if the
            ensemble is created from an optimization result or
            (sampling_chain, sampling_step) if the ensemble is created from a
            sampling result.
        ensemble_type:
            Type of ensemble: Ensemble (default), sample, or unprocessed_chain
            Samples are meant to be representative, ensembles can be any
            ensemble of parameters, and unprocessed chains still have burn-ins
        predictions:
            List of EnsemblePrediction objects
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
            if len(lower_bound) == 1:
                self.lower_bound = np.full((x_vectors.shape[0],), lower_bound)
            else:
                self.lower_bound = lower_bound
        self.upper_bound = np.full((self.n_x,), np.nan)
        if upper_bound is not None:
            if np.array(upper_bound).size == 1:
                self.upper_bound = np.full((x_vectors.shape[0],), upper_bound)
            else:
                self.upper_bound = upper_bound

        # handle parameter names
        if x_names is not None:
            self.x_names = x_names
        else:
            self.x_names = [f'x_{ix}' for ix in range(self.n_x)]

        # Do we have a representative sample or just random ensemble?
        self.ensemble_type = EnsembleType.ensemble
        if ensemble_type is not None:
            self.ensemble_type = ensemble_type

        # Do we have predictions for this ensemble?
        self.predictions = []
        if predictions is not None:
            self.predictions = predictions

    def __iter__(self):
        """
        __iter__ makes the instances of the class iterable objects, allowing to
        apply functions such as __dict__ to them.
        """
        yield X_VECTOR, self.x_vectors
        yield NX, self.n_x
        yield X_NAMES, self.x_names
        yield NVECTORS, self.n_vectors
        yield VECTOR_TAGS, self.vector_tags
        yield ENSEMBLE_TYPE, self.ensemble_type
        yield PREDICTIONS, self.predictions
        yield SUMMARY, self.summary
        yield LOWER_BOUND, self.lower_bound
        yield UPPER_BOUND, self.upper_bound

    def _map_parameters_by_objective(self,
                                     predictor: Callable,
                                     fill_in_value: float = np.nan):
        """
        The parameters of the ensemble don't need to have the same ordering as
        in the predictor. This functions maps them onto each other
        """
        # create short hands
        parameter_ids_objective = predictor.amici_objective.x_names
        parameter_ids_ensemble = self.x_names
        # map and fill if not found
        mapping = [
            parameter_ids_ensemble.index(parameter_id_objective)
            if parameter_id_objective in parameter_ids_ensemble
            else fill_in_value
            for parameter_id_objective in parameter_ids_objective
        ]

        return mapping

    def predict(self,
                predictor: Callable,
                prediction_id: str = None,
                sensi_orders: Tuple = (0,),
                fill_in_value: float = np.nan,
                mode: str = MODE_FUN):
        """
        Convenience function to run predictions for a full ensemble:
        User needs to hand over a predictor function and settings, then all
        results are grouped as EnsemblePrediction for the whole ensemble

        Parameters
        ----------
        predictor:
            Prediction function, e.g., an AmiciPredictor

        prediction_id:
            Identifier for the predictions

        sensi_orders:
            Specifies which sensitivities to compute, e.g. (0,1) -> fval, grad

        fill_in_value:
            If parameters are needed in the mapping, which are not found in the
            parameter source, it can make sense to fill them up with this
            default value in some cases (to be used with caution though).

        mode:
            Whether to compute function values or residuals.

        Returns
        -------
        result:
            EnsemblePrediction of the ensemble for the predictor function
        """
        # preallocate
        prediction_results = []

        # get the correct parameter mapping
        mapping = self._map_parameters_by_objective(
            predictor, fill_in_value=fill_in_value)

        for ix in range(self.n_vectors):
            x = self.x_vectors[mapping, ix]
            prediction_results.append(predictor(x, sensi_orders, mode))

        return EnsemblePrediction(predictor=predictor,
                                  prediction_id=prediction_id,
                                  prediction_results=prediction_results)

    def compute_summary(self,
                        percentiles_list: Sequence[int] = (5, 20, 80, 95)):
        """
        This function computes the mean, the median, the standard deviation
        and possibly percentiles for the parameters of the ensemble.
        Those summary results are added as a data member to the
        EnsemblePrediction object.

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
        Use ensemble mean and standard deviation to assess (in a rudimentary
        way) whether or not parameters are identifiable. Returns a dataframe
        with tuples, which specify whether or not the lower and the upper
        bounds are violated.

        Returns
        -------
        parameter_identifiability:
            DataFrame indicating parameter identifiability based on mean
            plus/minus standard deviations and parameter bounds
        """
        # Recompute the summary, maybe the ensemble objects has been changed.
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
            perc_list = [int(i_key[11:]) for i_key in self.summary.keys()
                         if i_key[0:4] == 'perc']
            perc_lower = [perc for perc in perc_list if perc < 50]
            perc_upper = [perc for perc in perc_list if perc > 50]

            # create dict of identifiability
            tmp_identifiability = {
                'parameterId': x_name,
                'lowerBound': lb,
                'upperBound': ub,
                'ensemble_mean': mean,
                'ensemble_std': std,
                'ensemble_median': median,
                'within lb: 1 std': lb < mean - std,
                'within ub: 1 std': ub > mean + std,
                'within lb: 2 std': lb < mean - 2 * std,
                'within ub: 2 std': ub > mean + 2 * std,
                'within lb: 3 std': lb < mean - 3 * std,
                'within ub: 3 std': ub > mean + 3 * std,
            }
            # handle percentiles
            for perc in perc_lower:
                tmp_identifiability[f'within lb: perc {perc}'] = \
                    lb < self.summary[f'{PERCENTILE} {perc}'][ix]
            for perc in perc_upper:
                tmp_identifiability[f'within ub: perc {perc}'] = \
                    ub > self.summary[f'{PERCENTILE} {perc}'][ix]

            parameter_identifiability.append(tmp_identifiability)

        # create DataFrame
        parameter_identifiability = pd.DataFrame(parameter_identifiability)
        parameter_identifiability.index = \
            parameter_identifiability['parameterId']

        return parameter_identifiability
