"""Validation intervals."""

import logging
from typing import Optional
from copy import deepcopy

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
        n_starts: Optional[int] = 1,
        optimizer: Optional[Optimizer] = None,
        engine: Optional[Engine] = None,
        lsq_objective: bool = False,
        return_significance: bool = True,
) -> float:
    """
    A Validation Interval for significance alpha is a confidence region/
    interval for a new validation experiment. [#Kreutz]_ et al.
    (This method per default returns the significance = 1-alpha!)

    The reasoning behind their approach is, that a validation data set
    is outside the validation interval, if fitting the full data set
    would lead to a fit $\theta_{new}$, that does not contain the old
    fit $\theta_{train}$ in their (Profile-Likelihood) based
    parameter-confidence intervals. (I.e. the old fit would be rejected by
    the fit of the full data.)

    This method returns the significance of the validation data set (where
    `result_full_data` is the objective function for fitting both data sets).
    I.e. the largest alpha, such that there is a validation region/interval
    such that the validation data set lies outside this Validation
    Interval with probability alpha. (If one is interested in the opposite,
    set `return_significance=False`.)

    Parameters
    ----------
    problem_full_data:
        pypesto.problem, such that the objective is the
        negative-log-likelihood of the training and validation data set.

    result_training_data:
        result object from the fitting of the training data set only.

    result_full_data
        pypesto.result object that contains the result of fitting
        training and validation data combined.

    n_starts
        number of starts for fitting the full data set
        (if result_full_data is not provided).

    optimizer:
        optimizer used for refitting the data (if result_full_data is not
        provided).

    engine
        engine for refitting (if result_full_data is not provided).

    lsq_objective:
        indicates if the objective of problem_full_data corresponds to a nllh
        (False), or a chi^2 value (True).
    return_significance:
        indicates, if the function should return the significance (True) (i.e.
        the  probability, that the new data set lies outside the Confidence
        Interval for the validation experiment, as given by the method), or
        the largest alpha, such that the validation experiment still lies
        within the Confidence Interval (False). I.e. alpha = 1-significance.


        .. [#Kreutz] Kreutz, Clemens, Raue, Andreas and Timmer, Jens.
                  “Likelihood based observability analysis and
                  confidence intervals for predictions of dynamic models”.
                  BMC Systems Biology 2012/12.
                  doi:10.1186/1752-0509-6-120

     """

    if (result_full_data is not None) and (optimizer is not None):
        raise UserWarning("optimizer will not be used, as a result object "
                          "for the full data set is provided.")

    # if result for full data is not provided: minimize
    if result_full_data is None:

        x_0 = result_training_data.optimize_result.get_for_key('x')

        # copy problem, in order to not change/overwrite x_guesses
        problem = deepcopy(problem_full_data)
        problem.set_x_guesses(x_0)

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
        logger.warning("Fitting of the full data set provided a worse fit "
                       "than the fit only considering the training data. "
                       "Consider rerunning the not handing over "
                       "result_full_data or running the fitting from the "
                       "best parameters found from the training data.")

    # compute the probability, that the validation data set is outside the CI
    # => survival function chi.sf
    if return_significance:
        if lsq_objective:
            return chi2.sf(nllh_new-nllh_old, 1)
        else:
            return chi2.sf(2*(nllh_new-nllh_old), 1)
    # compute the probability, that the validation data set is inside the CI
    # => cumulative density function chi.cdf
    else:
        if lsq_objective:
            return chi2.cdf(nllh_new-nllh_old, 1)
        else:
            return chi2.cdf(2*(nllh_new-nllh_old), 1)
