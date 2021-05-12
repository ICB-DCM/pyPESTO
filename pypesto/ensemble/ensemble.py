import logging
from functools import partial
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, Callable, Dict, List

from .. import Result
from ..engine import (
    Engine,
    MultiProcessEngine,
    MultiThreadEngine,
    SingleCoreEngine,
)
from ..predict import (
    PredictionConditionResult,
    PredictionResult,
)
from ..sample import geweke_test
from .task import EnsembleTask
from .constants import (PREDICTOR, PREDICTION_ID, PREDICTION_RESULTS,
                        PREDICTION_ARRAYS, PREDICTION_SUMMARY, OUTPUT,
                        OUTPUT_SENSI, TIMEPOINTS, X_VECTOR, NX, X_NAMES,
                        NVECTORS, VECTOR_TAGS, PREDICTIONS, MODE_FUN,
                        EnsembleType, ENSEMBLE_TYPE, MEAN, MEDIAN,
                        STANDARD_DEVIATION, SUMMARY, LOWER_BOUND,
                        UPPER_BOUND, get_percentile_label)

logger = logging.getLogger(__name__)


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
                summary[get_percentile_label(perc)] = \
                    np.percentile(tmp_array, perc, axis=-1)
            return summary

        # preallocate for results
        cond_lists = {MEAN: [], STANDARD_DEVIATION: [], MEDIAN: []}
        for perc in percentiles_list:
            cond_lists[get_percentile_label(perc)] = []

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
                        output_ids=current_cond.output_ids
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
        # Do we have a representative sample or just random ensemble?
        self.ensemble_type = EnsembleType.ensemble
        if ensemble_type is not None:
            self.ensemble_type = ensemble_type

        # handle parameter vectors and sizes
        self.x_vectors = x_vectors
        self.n_x = x_vectors.shape[0]
        self.n_vectors = x_vectors.shape[1]
        self.vector_tags = vector_tags
        self.summary = None

        # store bounds
        self.lower_bound = np.full((self.n_x,), np.nan)
        if lower_bound is not None:
            if np.array(lower_bound).size == 1:
                self.lower_bound = np.full((x_vectors.shape[0],), lower_bound)
            else:
                self.lower_bound = lower_bound
        self.upper_bound = np.full(self.n_x, np.nan)
        if upper_bound is not None:
            if np.array(upper_bound).size == 1:
                self.upper_bound = np.full(x_vectors.shape[0], upper_bound)
            else:
                self.upper_bound = upper_bound

        # handle parameter names
        if x_names is not None:
            self.x_names = x_names
        else:
            self.x_names = [f'x_{ix}' for ix in range(self.n_x)]

        # Do we have predictions for this ensemble?
        self.predictions = []
        if predictions is not None:
            self.predictions = predictions

    @staticmethod
    def from_sample(
            result: Result,
            remove_burn_in: bool = True,
            chain_slice: slice = None,
            **kwargs,
    ):
        """Construct an ensemble from a sample.

        Parameters
        ----------
        result:
            A pyPESTO result that contains a sample result.
        remove_burn_in:
            Exclude parameter vectors from the ensemble if they are in the
            "burn-in".
        chain_slice:
            Subset the chain with a slice. Any "burn-in" removal occurs first.

        Returns
        -------
        The ensemble.
        """
        x_vectors = result.sample_result.trace_x[0]
        if remove_burn_in:
            if result.sample_result.burn_in is None:
                geweke_test(result)
            burn_in = result.sample_result.burn_in
            x_vectors = x_vectors[burn_in:]
        if chain_slice is not None:
            x_vectors = x_vectors[chain_slice]
        x_vectors = x_vectors.T
        return Ensemble(x_vectors, **kwargs)

    @staticmethod
    def from_optimization_endpoints(
            result: Result,
            cutoff: float = np.inf,
            max_size: int = np.inf,
            **kwargs,
    ):
        """Construct an ensemble from an optimization result.

        Parameters
        ----------
        result:
            A pyPESTO result that contains an optimization result.
        cutoff:
            Exclude parameters from the optimization if the
            nllh is higher than the `cutoff`.
        max_size:
            The maximum size the ensemble should be.

        Returns
        -------
        The ensemble.
        """
        x_vectors = []
        vector_tags = []

        for start in result.optimize_result.list:
            # add the parameters from the next start as long as we
            # did not reach maximum size and the next value is still
            # lower than the cutoff value
            if start['fval'] <= cutoff and len(x_vectors) < max_size:
                x_vectors.append(start['x'])

                # the vector tag will be a -1 to indicate it is the last step
                vector_tags.append((int(start['id']), -1))
            else:
                break

        # print a warning if there are no vectors within the ensemble
        if len(x_vectors) == 0:
            raise ValueError('The ensemble does not contain any vectors. '
                             'Either the cutoff value was too small\n or the '
                             'result.optimize_result object might be empty.')
        elif len(x_vectors) < max_size:
            logger.info(f'The ensemble contains {len(x_vectors)} parameter '
                        'vectors, which is less than the maximum size.\nIf '
                        'you want to include more \nvectors, you can consider '
                        'raising the cutoff value or including parameters '
                        'from \nthe history with the `from_history` function.')

        x_vectors = np.stack(x_vectors, axis=1)
        return Ensemble(x_vectors=x_vectors,
                        x_names=result.problem.x_names,
                        vector_tags=vector_tags,
                        lower_bound=result.problem.lb_full,
                        upper_bound=result.problem.ub_full,
                        **kwargs)

    @staticmethod
    def from_optimization_history(
            result: Result,
            cutoff: float = np.inf,
            max_size: int = np.inf,
            max_per_start: int = np.inf,
            distribute: bool = True,
            **kwargs,
    ):
        """Construct an ensemble from the history of an optimization.

        Parameters
        ----------
        result:
            A pyPESTO result that contains an optimization result
            with history recorded.
        cutoff:
            Exclude parameters from the optimization if the nllh
            is higher than the `cutoff`.
        max_size:
            The maximum size the ensemble should be.
        max_per_start:
            The maximum number of vectors to be included from a
            single optimization start.
        distribute:
            Boolean flag, whether the best (False) values from the
            start should be taken or whether the indices should be
            more evenly distributed.

        Returns
        -------
        The ensemble.
        """
        if not result.optimize_result.list[0].history.options['trace_record']:
            logger.warning('The optimize result has no trace. The Ensemble '
                           'will automatically be created through '
                           'from_optimization_endpoints().')
            return Ensemble.from_optimization_endpoints(result=result,
                                                        cutoff=cutoff,
                                                        max_size=max_size,
                                                        **kwargs)
        x_vectors = []
        vector_tags = []
        x_names = result.problem.x_names
        lb = result.problem.lb_full
        ub = result.problem.ub_full

        # calculate the number of starts whose final nllh is below cutoff
        n_starts = sum(start['fval'] <= cutoff
                       for start in result.optimize_result.list)

        fval_trace = [
            np.array(
                result.optimize_result.list[i_ms]['history'].get_fval_trace()
            )
            for i_ms in range(n_starts)
        ]
        x_trace = [
            result.optimize_result.list[i_ms]['history'].get_x_trace()
            for i_ms in range(n_starts)
        ]

        # calculate the number of iterations included from each start
        n_per_starts = entries_per_start(fval_traces=fval_trace,
                                         cutoff=cutoff,
                                         max_per_start=max_per_start,
                                         max_size=max_size)
        # determine x_vectors from each start
        for start in range(n_starts):
            indices = get_vector_indices(trace_start=fval_trace[start],
                                         cutoff=cutoff,
                                         n_vectors=n_per_starts[start],
                                         distribute=distribute)
            x_vectors.extend([x_trace[start][ind] for ind in indices])
            vector_tags.extend([
                (int(result.optimize_result.list[start]['id']), ind)
                for ind in indices
            ])

        # raise a `ValueError` if there are no vectors within the ensemble
        if len(x_vectors) == 0:
            raise ValueError('The ensemble does not contain any vectors. '
                             'Either the `cutoff` value was too \nsmall '
                             'or the `result.optimize_result` object might '
                             'be empty.')

        x_vectors = np.stack(x_vectors, axis=1)
        return Ensemble(x_vectors=x_vectors,
                        x_names=x_names,
                        vector_tags=vector_tags,
                        lower_bound=lb,
                        upper_bound=ub,
                        **kwargs)

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

    def _map_parameters_by_objective(
            self,
            predictor: Callable,
            default_value: float = None,
    ):
        """
        The parameters of the ensemble don't need to have the same ordering as
        in the predictor. This functions maps them onto each other
        """
        # create short hands
        parameter_ids_objective = predictor.amici_objective.x_names
        parameter_ids_ensemble = self.x_names
        # map, and fill with `default_value` if not found and `default_value`
        # is specified.
        mapping = []
        for parameter_id_objective in parameter_ids_objective:
            if parameter_id_objective in parameter_ids_ensemble:
                mapping.append(
                    parameter_ids_ensemble.index(parameter_id_objective)
                )
            elif default_value is not None:
                mapping.append(default_value)
        return mapping

    def predict(
            self,
            predictor: Callable,
            prediction_id: str = None,
            sensi_orders: Tuple = (0,),
            default_value: float = None,
            mode: str = MODE_FUN,
            engine: Engine = None,
            progress_bar: bool = True
    ) -> EnsemblePrediction:
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

        default_value:
            If parameters are needed in the mapping, which are not found in the
            parameter source, it can make sense to fill them up with this
            default value (e.g. `np.nan`) in some cases (to be used with
            caution though).

        mode:
            Whether to compute function values or residuals.

        engine:
            Parallelization engine. Defaults to sequential execution on a
            `SingleCoreEngine`.

        progress_bar:
            Whether to display a progress bar.

        Returns
        -------
        The prediction of the ensemble.
        """
        if engine is None:
            engine = SingleCoreEngine()

        # Vectors are chunked to improve parallization performance.
        n_chunks = self.n_vectors  # Default is no chunking.
        if isinstance(engine, MultiProcessEngine):
            n_chunks = engine.n_procs
        if isinstance(engine, MultiThreadEngine):
            n_chunks = engine.n_threads
        chunks = [
            (
                (chunk_i+0) * int(np.floor(self.n_vectors / n_chunks)),
                (chunk_i+1) * int(np.floor(self.n_vectors / n_chunks)),
            )
            for chunk_i in range(n_chunks)
        ]
        # Last chunk should contain any remaining vectors that may have
        # been skipped due to the `floor` method.
        chunks[-1] = (chunks[-1][0], self.n_vectors)

        # Get the correct parameter mapping.
        mapping = self._map_parameters_by_objective(
            predictor,
            default_value=default_value,
        )

        # Setup the tasks with the prediction method and chunked vectors.
        method = partial(predictor, sensi_orders=sensi_orders, mode=mode)
        tasks = [
            EnsembleTask(
                method=method,
                vectors=self.x_vectors[mapping, chunk_start:chunk_end],
                id=chunk_i,
            )
            for chunk_i, (chunk_start, chunk_end) in enumerate(chunks)
        ]

        # Execute tasks and flatten chunked results.
        prediction_results = [
            prediction_result
            for prediction_chunk in engine.execute(
                tasks, progress_bar=progress_bar)
            for prediction_result in prediction_chunk
        ]

        return EnsemblePrediction(
            predictor=predictor,
            prediction_id=prediction_id,
            prediction_results=prediction_results,
        )

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
            summary[get_percentile_label(perc)] = \
                np.percentile(self.x_vectors, perc, axis=1)
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
                    lb < self.summary[get_percentile_label(perc)][ix]
            for perc in perc_upper:
                tmp_identifiability[f'within ub: perc {perc}'] = \
                    ub > self.summary[get_percentile_label(perc)][ix]

            parameter_identifiability.append(tmp_identifiability)

        # create DataFrame
        parameter_identifiability = pd.DataFrame(parameter_identifiability)
        parameter_identifiability.index = \
            parameter_identifiability['parameterId']

        return parameter_identifiability


def entries_per_start(fval_traces: List['np.ndarray'],
                      cutoff: float,
                      max_size: int,
                      max_per_start: int, ):
    """
    Creates the indices of each start that will be included
    in the ensemble.

    Parameters
    ----------
    fval_traces:
        the fval-trace of each start.
    cutoff:
        Exclude parameters from the optimization if the nllh
        is higher than the `cutoff`.
    max_size:
        The maximum size the ensemble should be.
    max_per_start:
        The maximum number of vectors to be included from a
        single optimization start.

    Returns
    -------
        A list of number of candidates per start that are to
        be included in the ensemble.

    """
    # choose possible candidates
    ens_ind = [np.flatnonzero(fval <= cutoff) for fval in fval_traces]

    # count the number of candidates per start
    n_per_start = np.array([len(start) for start in ens_ind])

    # if all possible indices can be included, return
    if (n_per_start < max_per_start).all() and sum(n_per_start) < max_size:
        return ens_ind

    # trimm down starts that exceed the limit:
    n_per_start = [min(n, max_per_start) for n in n_per_start]

    # trimm down more until it fits the max size
    decr = 0
    while(sum(n_per_start) > max_size):
        n_per_start = [min(n, max_per_start-decr)
                       for n in n_per_start]
        decr += 1
    # TODO: Possibility. With this implementation we could
    #  in a scenario, where we have more candidates than
    #  max size end up with an ensemble of size
    #  `max_size - len(n_starts)` in the worst case. We could introduce
    #  a flag which would be `force_max`, that indicates
    #  whether those remaining free slots should be filled by
    #  entries from certain starts. This would brng up the
    #  discussion which starts to choose. One obvious choice
    #  would be the best starts based on their endpoint.

    return n_per_start


def get_vector_indices(trace_start: np.ndarray,
                       cutoff: float,
                       n_vectors: int,
                       distribute: bool, ):
    """
    Returns the indices to be taken into an ensemble.

    Parameters
    ----------
    trace_start:
        The fval_trace of a single start.
    cutoff:
        Exclude parameters from the optimization if the nllh
        is higher than the `cutoff`.
    n_vectors:
        The number of indices to be included from one start.
    distribute:
        Boolean flag, whether the best (False) values from the
        start should be taken or whether the indices should be
        more evenly distributed.

    Returns
    -------
        The indices to include in the ensemble.
    """

    candidates = np.flatnonzero(trace_start <= cutoff)

    if distribute:
        indices = np.round(np.linspace(0, len(candidates) - 1, n_vectors))
        return candidates[indices.astype(int)]
    else:
        return candidates[:n_vectors]
