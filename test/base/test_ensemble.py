import numpy as np
import pypesto
import os
from pypesto.ensemble import (Ensemble,
                              write_ensemble_prediction_to_h5,
                              read_ensemble_prediction_from_h5,)
import scipy.optimize as so

from ..visualize import create_petab_problem
from functools import partial
from pypesto.predict.constants import (AMICI_STATUS,
                                       AMICI_T,
                                       AMICI_Y)
from pypesto.ensemble.constants import MEAN, WEIGHTED_SIGMA
from pypesto.predict import AmiciPredictor
from pypesto.engine import MultiProcessEngine


def test_ensemble_from_optimization():
    """
    Test reading an ensemble from optimization result.
    """
    objective = pypesto.Objective(fun=so.rosen,
                                  grad=so.rosen_der,
                                  hess=so.rosen_hess)
    dim_full = 10
    lb = -5 * np.ones((dim_full, 1))
    ub = 5 * np.ones((dim_full, 1))
    n_starts = 5

    problem = pypesto.Problem(objective=objective, lb=lb, ub=ub)

    optimizer = pypesto.optimize.ScipyOptimizer(options={'maxiter': 10})
    history_options = pypesto.HistoryOptions(trace_record=True)
    result = pypesto.optimize.minimize(
        problem=problem, optimizer=optimizer,
        n_starts=n_starts, history_options=history_options)

    # change fvals of each start
    for i_start, optimizer_result in enumerate(result.optimize_result.list):
        optimizer_result['fval'] = i_start+1
        for i_iter in range(len(optimizer_result['history']._trace['fval'])):
            optimizer_result['history']._trace['fval'][i_iter] = \
                len(optimizer_result['history']._trace['fval'])+i_start-i_iter

    # test_endpoints
    ensemble_ep = Ensemble.from_optimization_endpoints(
        result=result, cutoff=4, max_size=10
    )

    ensemble_hist = Ensemble.from_optimization_history(
        result=result, cutoff=4, max_size=10, max_per_start=5
    )

    # compare vector_tags with the expected values:
    ep_tags = [(int(result.optimize_result.list[i]['id']), -1)
               for i in [0, 1, 2, 3]]

    hist_tags = [
        (int(result.optimize_result.list[i]['id']),
         len(result.optimize_result.list[i]['history']._trace['fval'])-1-j)
        for i in range(4) for j in reversed(range(4-i))
    ]
    assert hist_tags == ensemble_hist.vector_tags
    assert ep_tags == ensemble_ep.vector_tags


def test_ensemble_prediction_from_hdf5():
    """
    Test writing an ensemble prediction to hdf5 and loading it.
    """
    problem = create_petab_problem()

    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            amici_output[output_type] if amici_output[AMICI_STATUS] == 0
            else np.full((len(amici_output[AMICI_T]), len(output_ids)), np.nan)
            for amici_output in amici_outputs
        ]
        return outputs
    post_processor_y = partial(
        post_processor,
        output_type=AMICI_Y,
        output_ids=problem.objective.amici_model.getObservableIds(),
    )
    predictor_y = AmiciPredictor(
        problem.objective,
        post_processor=post_processor_y,
        output_ids=problem.objective.amici_model.getObservableIds(),
    )
    ensemble_prediction = get_ensemble_prediction(max_size=10)

    fn = 'test_file.hdf5'
    try:
        write_ensemble_prediction_to_h5(ensemble_prediction, fn)
        ensemble_prediction_r = \
            read_ensemble_prediction_from_h5(predictor_y, fn)

        # test both Ensemble.Predictions
        assert ensemble_prediction.prediction_id ==\
               ensemble_prediction_r.prediction_id
        for i_run, _ in enumerate(
                ensemble_prediction.prediction_results):
            assert ensemble_prediction.prediction_results[i_run] == \
                   ensemble_prediction_r.prediction_results[i_run]
    finally:
        if os.path.exists(fn):
            os.remove(fn)


def test_ensemble_weighted_trajectory():
    """
    Test computing a weighted mean trajectory
    and weighted sigmas for an ensemble.
    """
    ens_pred = get_ensemble_prediction(inc_weights=True,
                                       inc_sigmay=True)

    # artifically set the values
    ens_pred.prediction_results[0].conditions[0].output = \
        np.ones((10, 1))
    ens_pred.prediction_results[1].conditions[0].output = \
        2 * np.ones((10, 1))
    ens_pred.prediction_results[0].conditions[0].output_weight \
        = np.log(10)
    ens_pred.prediction_results[1].conditions[0].output_weight \
        = np.log(20)
    ens_pred.prediction_results[0].conditions[0].output_sigmay \
        = 0.5 * np.ones((10, 1))
    ens_pred.prediction_results[1].conditions[0].output_sigmay \
        = 0.1 * np.ones((10, 1))

    ens_pred.compute_summary(weighting=True,
                             compute_weighted_sigma=True)

    np.testing.assert_almost_equal(
        ens_pred.prediction_summary[MEAN].conditions[0].output,
        5/3 * np.ones((10, 1)),
        decimal=15
    )
    np.testing.assert_almost_equal(
        ens_pred.prediction_summary[WEIGHTED_SIGMA].conditions[0].output,
        0.3 * np.ones((10, 1)),
        decimal=15
    )


def get_ensemble_prediction(max_size: int = 2,
                            inc_weights: bool = False,
                            inc_sigmay: bool = False):
    """
    Creates an ensemble prediction for the tests.
    """
    problem = create_petab_problem()

    optimizer = pypesto.optimize.ScipyOptimizer()
    result = pypesto.optimize.minimize(
        problem=problem, optimizer=optimizer,
        n_starts=2, )

    ensemble_ep = Ensemble.from_optimization_endpoints(
        result=result, max_size=10
    )

    # This post_processor will transform the output of the simulation tool
    # such that the output is compatible with the next steps.
    def post_processor(amici_outputs, output_type, output_ids):
        outputs = [
            amici_output[output_type] if amici_output[AMICI_STATUS] == 0
            else np.full((len(amici_output[AMICI_T]), len(output_ids)), np.nan)
            for amici_output in amici_outputs
        ]
        return outputs

    amici_objective = result.problem.objective
    observable_ids = amici_objective.amici_model.getObservableIds()
    post_processor_y = partial(
        post_processor,
        output_type=AMICI_Y,
        output_ids=observable_ids,
    )
    # Create pyPESTO predictors for states and observables
    predictor_y = AmiciPredictor(
        amici_objective,
        post_processor=post_processor_y,
        output_ids=observable_ids,
    )
    engine = MultiProcessEngine()
    ensemble_prediction = ensemble_ep.predict(predictor_y,
                                              prediction_id=AMICI_Y,
                                              engine=engine,
                                              include_llh_weights=inc_weights,
                                              include_sigmay=inc_sigmay)
    return ensemble_prediction
