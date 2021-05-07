import logging
from typing import Optional

from ..engine import Engine
from ..optimize import Optimizer, minimize
from ..problem import Problem
from ..result import Result
from scipy.stats import chi2


logger = logging.getLogger(__name__)


def validation_profile_significance(
        problem_full_data: Problem,
        result_training_data: Result,
        result_full_data: Optional[Result] = None,
        n_starts: int = 1,
        optimizer: Optional[Optimizer] = None,
        engine: Engine = None,
        lsq_objective: float = False
) -> float:
    """
    Computes the significance of an validation experiment as described in
    Kreutz et al. BMC Systems Biology 2012.


    Parameters
    ----------
    problem_full_data:
        pypesto.problem, such that the objective is the
        negative-log-likelihood of the training and validation data set.

    result_training_data:
        result object from the fitting of the training data set only.

    result_full_data
        pypesto.result object, that either contains, or (if fitting was not
        done beforehand) stores the parameter fitting of training and
        validation data combined.

    n_starts
        number of starts for fitting the full data set.

    optimizer:
        optimizer used for refitting the data (if result_full_data is not
        provided)

    engine
        engine for refitting.

    lsq_objective:
        indicates, if the objective of problem_full_data corresponds to a nllh
        (False), or a chi^2 value (True).
    """

    if (result_full_data is not None) and (optimizer is not None):
        raise UserWarning("optimizer will not be used, as a result object "
                          "for the full data set is provided.")

    if result_full_data is None:

        x_0 = result_training_data.optimize_result.get_for_key('x')

        # copy problem, in order to not change/overwrite x_guesses
        problem = Problem(
            objective=problem_full_data.objective,
            lb=problem_full_data.lb,
            ub=problem_full_data.ub,
            dim_full=problem_full_data.dim_full,
            x_fixed_indices=problem_full_data.x_fixed_indices,
            x_fixed_vals=problem_full_data.x_fixed_vals,
            x_guesses=x_0,
            startpoint_method=problem_full_data.startpoint_method,
            x_names=problem_full_data.x_names,
            x_scales=problem_full_data.x_scales,
            x_priors_defs=problem_full_data.x_priors,
            lb_init=problem_full_data.lb_init,
            ub_init=problem_full_data.ub_init)

        result_full_data = minimize(problem=problem,
                                    optimizer=optimizer,
                                    n_starts=n_starts,
                                    engine=engine)

    # Validation intervals compare the nllh value on the full data set
    # of the parameter fits from the training and the full data set.

    nllh_new = \
        result_full_data.optimize_result.get_for_key('fval')[0]
    nllh_old = \
        problem_full_data.objective(
            problem_full_data.get_reduced_vector(
                result_training_data.optimize_result.get_for_key('x')[0]))

    if nllh_new > nllh_old:
        raise RuntimeError("Fitting of the full data set provided a worse fit "
                           "then the fit only considering the training data. "
                           "Consider rerunning the not handing over "
                           "result_full_data or running the fitting from the "
                           "best parameters found from the training data.")

    if lsq_objective:
        return chi2.sf(0.5*(nllh_new-nllh_old), 1)
    else:
        return chi2.sf(nllh_new-nllh_old, 1)
