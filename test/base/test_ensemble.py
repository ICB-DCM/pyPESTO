import numpy as np
import pypesto
from pypesto.ensemble import Ensemble
import scipy.optimize as so


def test_ensemble_from_optimization():
    """
    Test, whether reading ensemble from history works as intended.
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
